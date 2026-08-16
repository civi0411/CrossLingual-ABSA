"""
src/data/dataset.py
───────────────────
Data cleaning utilities and two distinct Dataset classes for sentence-level ABSA tasks.

Classes:
    ABSADataset: Used for L1 (AG-CAN) and L2 (XLM-R) classification models.
    ABSASeq2SeqDataset: Used for L3 (mT5) sequence-to-sequence generation models.

Both datasets standardize labels to the integer set {0, 1, 2} corresponding to
{positive, negative, neutral} for unified evaluation across architectures.

Key Design Decisions:
    - `category_mask` is constructed at the dataset level rather than inside the model.
      This ensures mathematically exact masked mean pooling for XLM-R without positional slicing.
    - `dual_encode` flag separates L1 (two distinct tensors: text and aspect) from L2
      (single concatenated tensor: text + category).
    - Unicode NFC normalization is strictly required for accurate handling of Vietnamese
      diacritics and German Umlauts.
    - No lowercasing and no stopword removal are performed during text cleaning to preserve
      case-sensitive features (e.g., German nouns, XLM-R capitalization distinctions).
    - Implicit aspects are detected via `term == "NULL"`.
"""

import re, json, unicodedata, logging
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

log = logging.getLogger(__name__)

LABEL2ID  = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL  = {0: "positive", 1: "negative", 2: "neutral"}
TEXT2LABEL = ID2LABEL   # alias for decoding


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    Apply Unicode NFC normalization and clean redundant whitespace.

    NFC normalization is mandatory for languages with diacritics and composite characters
    such as Vietnamese tone marks and German Umlauts. Text is not lowercased to preserve
    case distinctions required by XLM-R and German noun capitalization. Punctuation is retained
    as it conveys sentiment intensity.

    Args:
        text (str): Raw input text string.

    Returns:
        str: Cleaned and NFC-normalized text string.
    """
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_and_validate(
    samples: list[dict],
    valid_categories: set[str],
    tokenizer=None,
    max_len: int = 128,
    min_tokens: int = 3,
    log_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Perform two-layer validation and cleaning on raw dataset samples.

    Validation layers:
        - Layer 1 (Text Quality): Apply NFC normalization and enforce minimum token count.
        - Layer 2 (Schema Validation): Ensure valid sentiment labels and known categories.

    Also ensures the `is_implicit` boolean field is populated based on whether `term == "NULL"`.
    Note that token-level length checks are omitted here because sequence-to-sequence prompts
    alter token lengths; truncation is applied during tokenization in each Dataset class.

    Args:
        samples (list[dict]): List of raw sample dictionaries.
        valid_categories (set[str]): Set of valid aspect categories extracted from source training data.
        tokenizer (Optional[Any]): Tokenizer instance (unused in validation logic, kept for API compatibility).
        max_len (int): Maximum sequence length threshold. Defaults to 128.
        min_tokens (int): Minimum word token count required for valid text. Defaults to 3.
        log_path (Optional[str]): File path to write dropped sample logs. Defaults to None.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing (`clean_samples`, `dropped_log`).
    """
    clean, dropped = [], []

    for s in samples:
        # Normalize text and term
        s["text"] = normalize_text(s["text"])
        if "term" in s:
            s["term"] = normalize_text(s["term"])

        # Ensure is_implicit field (if not present, infer from term)
        if "is_implicit" not in s:
            s["is_implicit"] = (s.get("term") == "NULL")

        # L1: text length check
        if not s["text"] or len(s["text"].split()) < min_tokens:
            dropped.append({"id": s["id"], "reason": "too_short", "lang": s["lang"]})
            continue

        # L2: label and category validation
        if s["sentiment"] not in LABEL2ID:
            dropped.append({"id": s["id"], "reason": f"bad_label:{s['sentiment']}", "lang": s["lang"]})
            continue
        if s["category"] not in valid_categories:
            dropped.append({"id": s["id"], "reason": f"unknown_cat:{s['category']}", "lang": s["lang"]})
            continue

        # (L3 token length check removed – seq2seq prompt makes it misleading;
        #  truncation is applied during tokenization in each Dataset class.)

        clean.append(s)

    by_lang = defaultdict(int)
    for d in dropped:
        by_lang[d["lang"]] += 1
    log.info("Clean: %d kept, %d dropped %s", len(clean), len(dropped), dict(by_lang))

    if log_path and dropped:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            for d in dropped:
                f.write(json.dumps(d) + "\n")

    return clean, dropped


def build_valid_categories(en_samples: list[dict]) -> set[str]:
    """
    Extract the set of dynamic aspect categories from English training samples.

    Args:
        en_samples (list[dict]): List of English training sample dictionaries.

    Returns:
        set[str]: Set of unique category names found in the samples.
    """
    return {s["category"] for s in en_samples}


# ══════════════════════════════════════════════════════════════════════════════
#  L1 + L2: Classification Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ABSADataset(Dataset):
    """
    PyTorch Dataset for sentence-pair aspect category sentiment classification.

    This dataset supports two encoding paradigms governed by `dual_encode`:
        - When `dual_encode=False` (L2 XLM-R):
          Constructs a single merged sequence: `[CLS] text [SEP] category [SEP]`.
          Also generates a binary `category_mask` with values set to 1 at category token positions
          and 0 elsewhere, enabling mathematically exact masked mean pooling in XLM-R.
        - When `dual_encode=True` (L1 AG-CAN):
          Constructs two separate tensor encodings: `text_ids` and `aspect_ids`.
          `text_ids` encode the input sentence while `aspect_ids` encode the aspect category
          to serve as query embeddings in Aspect-Guided Multi-Head Attention.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_len: int = 128,
        dual_encode: bool = False,
    ):
        self.dual   = dual_encode
        self.labels = []
        self.meta   = []

        texts = [s["text"]     for s in samples]
        cats  = [s["category"] for s in samples]

        for s in samples:
            self.labels.append(s["label"])
            self.meta.append({
                "id": s["id"], "lang": s["lang"],
                "domain": s["domain"], "category": s["category"],
                "is_implicit": s["is_implicit"], "sentiment": s["sentiment"],
                "text": s["text"],
            })

        if dual_encode:
            # L1: encode text and category SEPARATELY
            self.text_enc = tokenizer(
                texts, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            self.asp_enc = tokenizer(
                cats, max_length=32, truncation=True,
                padding="max_length", return_tensors="pt",
            )
        else:
            # L2: sentence pair encoding
            batch_enc = tokenizer(
                texts, cats, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            self.enc = batch_enc

            # Build category_mask for exact masked mean pooling
            # Use token_type_ids if available; otherwise build manually with SEP detection
            if "token_type_ids" in batch_enc:
                self.cat_mask = batch_enc["token_type_ids"]
            else:
                # XLM-R has no token_type_ids — build mask using SEP token positions
                self.cat_mask = self._build_category_mask_safe(
                    batch_enc["input_ids"], tokenizer, cats
                )

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _build_category_mask_safe(self, input_ids, tokenizer, categories):
        """
        Construct a category token mask for XLM-R when `token_type_ids` are unavailable.

        Searches for subword token IDs of the aspect category strictly after the first
        `[SEP]` token (`sep_token_id`) to avoid false positive matches within the review text.

        Args:
            input_ids (torch.Tensor): Batch of tokenized sequence IDs of shape (batch_size, seq_len).
            tokenizer (PreTrainedTokenizer): Tokenizer instance used to encode aspect categories.
            categories (list[str]): List of aspect category strings corresponding to each sequence.

        Returns:
            torch.Tensor: Binary category mask tensor of shape (batch_size, seq_len).
        """
        masks = []
        sep_id = tokenizer.sep_token_id
        for ids, cat in zip(input_ids, categories):
            cat_ids = tokenizer.encode(cat, add_special_tokens=False)
            id_list = ids.tolist()
            mask = [0] * len(id_list)

            # Find first [SEP] position
            try:
                sep_pos = id_list.index(sep_id)
            except ValueError:
                sep_pos = 0  # fallback, should not happen

            # Search for cat_ids only after sep_pos
            match_found = False
            for i in range(sep_pos + 1, len(id_list) - len(cat_ids) + 1):
                if id_list[i:i+len(cat_ids)] == cat_ids:
                    for j in range(len(cat_ids)):
                        mask[i+j] = 1
                    match_found = True
                    break
                    
            if not match_found:
                log.warning("Category '%s' not found in tokens after SEP. Possibly truncated. Aspect vector will be zero.", cat)

            masks.append(mask)
        return torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.dual:
            return {
                "input_ids":       self.text_enc["input_ids"][idx],
                "attention_mask":  self.text_enc["attention_mask"][idx],
                "aspect_ids":      self.asp_enc["input_ids"][idx],
                "aspect_mask":     self.asp_enc["attention_mask"][idx],
                "labels":          self.labels[idx],
                "meta":            self.meta[idx],
            }
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "category_mask":  self.cat_mask[idx],
            "labels":         self.labels[idx],
            "meta":           self.meta[idx],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  L3: Seq2Seq Dataset
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = "aspect: {category} review: {text}"


class ABSASeq2SeqDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-sequence aspect sentiment generation using mT5.

    Input prompt format : `"aspect: {category} review: {text}"`
    Target output format: `"positive" / "negative" / "neutral"`
    Post-processing     : Generated strings are mapped back to integer IDs {0, 1, 2}.

    Implicit aspects (`term == "NULL"`) are handled naturally without special tokens,
    allowing the generative model to infer sentiment directly from context and category.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_input_len: int = 128,
        max_target_len: int = 10,
    ):
        self.meta       = []
        self.ref_labels = []
        inputs, targets = [], []

        for s in samples:
            inputs.append(PROMPT.format(text=s["text"], category=s["category"]))
            targets.append(ID2LABEL[s["label"]])
            self.ref_labels.append(s["label"])
            self.meta.append({
                "id": s["id"], "lang": s["lang"], "domain": s["domain"],
                "category": s["category"], "is_implicit": s["is_implicit"],
                "sentiment": s["sentiment"], "label": s["label"],
                "text": s["text"],
            })

        self.input_enc = tokenizer(
            inputs, max_length=max_input_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        tgt_enc = tokenizer(
            text_target=targets, max_length=max_target_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        lbl = tgt_enc["input_ids"].clone()
        lbl[lbl == tokenizer.pad_token_id] = -100
        self.target_labels = lbl
        self.ref_labels    = torch.tensor(self.ref_labels, dtype=torch.long)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_enc["input_ids"][idx],
            "attention_mask": self.input_enc["attention_mask"][idx],
            "labels":         self.target_labels[idx],
            "ref_labels":     self.ref_labels[idx],
            "meta":           self.meta[idx],
        }

    @staticmethod
    def decode_to_labels(generated_ids, tokenizer) -> list[int]:
        """
        Decode generated token IDs into integer label indices {0, 1, 2}.

        Defaults to index 2 (`neutral`) if the decoded text does not match any valid label string.
        Note: Lowercasing is not applied here because mT5 outputs lowercase strings directly.
        This preserves exact string matching with training target formatting.

        Args:
            generated_ids (torch.Tensor | list[list[int]]): Sequence of output token IDs from model generation.
            tokenizer (PreTrainedTokenizer): Tokenizer instance to decode token IDs.

        Returns:
            list[int]: List of predicted integer labels {0, 1, 2}.
        """
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # Direct lookup without lowercasing (mT5 produces lower case)
        return [LABEL2ID.get(t.strip(), 2) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════

def _collate(batch):
    meta   = [b.pop("meta") for b in batch]
    result = default_collate(batch)
    result["meta"] = meta
    return result


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate,
    )


def _prepare_data_setup(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    tokenizer,
    cfg: dict,
) -> tuple[int, list[dict], list[dict], list[dict]]:
    """
    Helper function to extract hyperparameters and clean dataset splits for DataLoader construction.

    Args:
        train_samples (list[dict]): Raw training samples.
        val_samples (list[dict]): Raw validation samples.
        test_samples (list[dict]): Raw test samples.
        tokenizer (PreTrainedTokenizer): Tokenizer instance used for validation.
        cfg (dict): Global configuration dictionary.

    Returns:
        tuple[int, list[dict], list[dict], list[dict]]: Tuple containing `max_len` and cleaned splits.
    """
    domain = train_samples[0]["domain"] if train_samples else "restaurant"
    max_len = cfg.get("max_len", {}).get(domain, 128)
    min_tokens = cfg.get("min_tokens", 3)

    valid_categories = build_valid_categories(train_samples)

    clean_train, _ = clean_and_validate(train_samples, valid_categories, tokenizer, max_len=max_len, min_tokens=min_tokens)
    clean_val, _ = clean_and_validate(val_samples, valid_categories, tokenizer, max_len=max_len, min_tokens=min_tokens)
    clean_test, _ = clean_and_validate(test_samples, valid_categories, tokenizer, max_len=max_len, min_tokens=min_tokens)
    
    return max_len, clean_train, clean_val, clean_test


def build_cls_loaders(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    cfg: dict,
    model_name: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    paths = cfg.get("model_paths", {})
    if model_name == "ag_can":
        model_path = paths.get("mbert", "pretrained_models/mbert")
        if not Path(model_path).exists():
            model_path = "bert-base-multilingual-cased"
    else:
        model_path = paths.get("xlmr", "pretrained_models/xlmr")
        if not Path(model_path).exists():
            model_path = "xlm-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    max_len, clean_train, clean_val, clean_test = _prepare_data_setup(
        train_samples, val_samples, test_samples, tokenizer, cfg
    )

    dual_encode = (model_name == "ag_can")
    train_ds = ABSADataset(clean_train, tokenizer, max_len=max_len, dual_encode=dual_encode)
    val_ds = ABSADataset(clean_val, tokenizer, max_len=max_len, dual_encode=dual_encode)
    test_ds = ABSADataset(clean_test, tokenizer, max_len=max_len, dual_encode=dual_encode)

    batch_size = cfg.get("batch_size", 16)
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_gen_loaders(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    tokenizer,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    max_target_len = cfg.get("max_target_len", 10)

    max_len, clean_train, clean_val, clean_test = _prepare_data_setup(
        train_samples, val_samples, test_samples, tokenizer, cfg
    )

    train_ds = ABSASeq2SeqDataset(clean_train, tokenizer, max_input_len=max_len, max_target_len=max_target_len)
    val_ds = ABSASeq2SeqDataset(clean_val, tokenizer, max_input_len=max_len, max_target_len=max_target_len)
    test_ds = ABSASeq2SeqDataset(clean_test, tokenizer, max_input_len=max_len, max_target_len=max_target_len)

    batch_size = cfg.get("batch_size", 16)
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
