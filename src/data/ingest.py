"""
src/data/ingest.py
──────────────────
Data ingestion pipeline that loads M-ABSA from HuggingFace (or local cache), explodes triplets
into individual sample dictionaries, and persists them as partitioned JSONL files.

Output structure: `data/processed/{domain}/{lang}/{split}.jsonl`

Key Design Decisions:
    - Aspect category strings are NOT lowercased (`"Camera#General"` is preserved exactly).
    - Sentiment labels are converted to lowercase to match `LABEL2ID` (`"positive"`, `"negative"`, `"neutral"`).
    - Implicit aspects (`term == "NULL"`) are automatically identified and flagged (`is_implicit=True`).
    - Every sample is assigned a globally unique ID: `{domain}_{lang}_{split}_{sent_idx}_{triplet_idx}`.

Strict data leakage guards (`assert_no_leakage` and `assert_test_not_seen`) are checked
during dataset construction in `build_datasets()`.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

from src.utils.common import ensure_dir

log = logging.getLogger(__name__)

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
HF_SPLIT_MAP = {"training": "training", "validation": "val", "test": "test"}


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: explode a single M-ABSA row into samples
# ══════════════════════════════════════════════════════════════════════════════

def _explode_row(row: dict, domain: str, hf_split: str, sent_idx: int) -> list[dict]:
    """
    Explode a raw M-ABSA dataset row (sentence and multiple triplets) into independent sample records.

    Each annotated triplet `(term, category, sentiment)` becomes a standalone sample record.

    Args:
        row (dict): Raw dataset dictionary containing `"sentence"`, `"lang"`, and `"triplets"`.
        domain (str): Domain identifier (`"restaurant"` or `"phone"`).
        hf_split (str): Raw HuggingFace split name (`"training"`, `"validation"`, or `"test"`).
        sent_idx (int): Zero-based sentence index within the original split.

    Returns:
        list[dict]: List of sample dictionaries containing keys: `id`, `text`, `term`,
            `category`, `sentiment`, `label`, `lang`, `domain`, `split`, and `is_implicit`.
    """
    split = HF_SPLIT_MAP[hf_split]          # "training"/"val"/"test"
    lang = row["lang"]
    text = row["sentence"].strip()
    samples = []

    for trip_idx, triplet in enumerate(row.get("triplets", [])):
        if len(triplet) != 3:
            continue          # malformed triplet, skip
        term, category, sentiment = triplet

        # Normalize term and sentiment
        term = term.strip()
        sentiment = sentiment.strip().lower()
        category = category.strip()          # Do NOT lower! Keep original case

        # Skip if invalid sentiment or empty text
        if sentiment not in LABEL2ID or not text:
            continue

        samples.append({
            "id": f"{domain}_{lang}_{split}_{sent_idx}_{trip_idx}",
            "text": text,
            "term": term if term else "NULL",
            "category": category,
            "sentiment": sentiment,
            "label": LABEL2ID[sentiment],
            "lang": lang,
            "domain": domain,
            "split": split,
            "is_implicit": (not term or term == "NULL"),
        })
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Load a complete domain from HuggingFace and save to JSONL
# ══════════════════════════════════════════════════════════════════════════════

def load_domain(domain: str, processed_dir: str) -> dict[str, list[dict]]:
    """
    Load an entire domain dataset from HuggingFace or local raw files, explode triplets, and write to JSONL.

    Prioritizes local raw files under `data/raw/{domain}/{lang}/` if present before falling back
    to the HuggingFace `Multilingual-NLP/M-ABSA` dataset repository.

    Args:
        domain (str): Domain name (`"restaurant"` or `"phone"`).
        processed_dir (str): Root directory path for processed JSONL outputs.

    Returns:
        dict[str, list[dict]]: Mapping from partition key `"{lang}_{split}"` to lists of sample dictionaries.
    """
    out_dir = ensure_dir(processed_dir)
    result = {}

    # Prioritize loading from local raw files if they exist
    raw_domain_dir = Path("data/raw") / domain
    if raw_domain_dir.exists():
        log.info("Loading domain '%s' from local raw files: %s", domain, raw_domain_dir)
        local_splits = {
            "train.txt": "training",
            "dev.txt": "validation",
            "test.txt": "test"
        }
        
        for lang_dir in sorted(raw_domain_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            
            for txt_file, hf_split in local_splits.items():
                file_path = lang_dir / txt_file
                if not file_path.exists():
                    continue
                    
                samples = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line or "####" not in line:
                            continue
                        text, blob = line.split("####", 1)
                        text = text.strip()
                        
                        import ast
                        try:
                            triplets = ast.literal_eval(blob)
                        except Exception:
                            import re
                            triplets = re.findall(r"\[\s*['\"](.*?)['\"]\s*,\s*['\"](.*?)['\"]\s*,\s*['\"](positive|negative|neutral)['\"]\s*\]", blob)
                        
                        row = {
                            "sentence": text,
                            "lang": lang,
                            "triplets": triplets
                        }
                        samples.extend(_explode_row(row, domain, hf_split, i))
                
                split = HF_SPLIT_MAP[hf_split]
                key = f"{lang}_{split}"
                
                if key not in result:
                    result[key] = []
                result[key].extend(samples)
                
                lang_dir_out = out_dir / domain / lang
                lang_dir_out.mkdir(parents=True, exist_ok=True)
                out_path = lang_dir_out / f"{split}.jsonl"
                with open(out_path, "w", encoding="utf-8") as out_f:
                    for s in samples:
                        out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                log.info("  %s/%s/%s (local) → %d samples", domain, lang, split, len(samples))
        
        return result

    # Fallback to HuggingFace
    log.info("Loading domain '%s' from HuggingFace...", domain)
    from datasets import load_dataset
    ds = load_dataset("Multilingual-NLP/M-ABSA", domain)

    for hf_split in ["training", "validation", "test"]:
        if hf_split not in ds:
            continue
        by_lang = defaultdict(list)

        for i, row in enumerate(ds[hf_split]):
            for sample in _explode_row(row, domain, hf_split, i):
                by_lang[sample["lang"]].append(sample)

        split = HF_SPLIT_MAP[hf_split]
        for lang, samples in by_lang.items():
            key = f"{lang}_{split}"
            lang_dir_out = out_dir / domain / lang
            lang_dir_out.mkdir(parents=True, exist_ok=True)
            out_path = lang_dir_out / f"{split}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            result[key] = samples
            log.info("  %s/%s/%s → %d samples", domain, lang, split, len(samples))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Load JSONL files (for later use)
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file from disk and return a list of parsed dictionary records."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(domain: str, lang: str, split: str, processed_dir: str) -> list[dict]:
    """Load pre-processed records for a specific partition (`domain`, `lang`, `split`)."""
    path = Path(processed_dir) / domain / lang / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run 'python -m src.data.ingest' first to generate it."
        )
    return load_jsonl(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Leakage guards
# ══════════════════════════════════════════════════════════════════════════════

def assert_no_leakage(
    train_samples: list[dict],
    allowed_langs: list[str],
    setting: str,
) -> None:
    """
    Strictly verify that the training set contains only languages specified in `allowed_langs`.

    Raises an `AssertionError` if any sample belonging to an unauthorized language is detected,
    preventing cross-lingual evaluation leakage.

    Args:
        train_samples (list[dict]): List of samples allocated for training.
        allowed_langs (list[str]): List of language codes permitted in the training partition.
        setting (str): Current experimental transfer setting identifier (`"s1"`, `"s2"`, or `"s3"`).
    """
    illegal = [s for s in train_samples if s["lang"] not in allowed_langs]
    if illegal:
        raise AssertionError(
            f"\n{'='*60}\n"
            f"DATA LEAKAGE in setting '{setting}'\n"
            f"Allowed langs : {allowed_langs}\n"
            f"Illegal langs : {set(s['lang'] for s in illegal)}\n"
            f"Count         : {len(illegal)}\n"
            f"Example IDs   : {[s['id'] for s in illegal[:3]]}\n"
            f"{'='*60}"
        )
    log.info(
        "Leakage PASSED [%s] — %d samples, langs=%s",
        setting, len(train_samples),
        set(s["lang"] for s in train_samples),
    )


def assert_test_not_seen(train: list[dict], test: list[dict]) -> None:
    """Ensure zero overlap between training and testing sample IDs to prevent test evaluation contamination."""
    train_ids = {s["id"] for s in train}
    overlap = [s for s in test if s["id"] in train_ids]
    if overlap:
        raise AssertionError(
            f"TEST LEAKAGE: {len(overlap)} test IDs found in training set.\n"
            f"Example: {overlap[0]['id']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Few‑shot stratified sampling
# ══════════════════════════════════════════════════════════════════════════════

def stratified_sample(samples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Draw exactly `n` samples using stratified sampling balanced across sentiment classes.

    If any sentiment class (`positive`, `negative`, `neutral`) has fewer items than its allocated quota,
    the shortfall is redistributed across the remaining classes. If `n >= len(samples)`, returns a full copy.

    Args:
        samples (list[dict]): Pool of candidate sample dictionaries.
        n (int): Total number of samples to select (`n_shot`).
        seed (int): Random seed for reproducible stratification. Defaults to 42.

    Returns:
        list[dict]: Stratified subset containing `n` sample records.
    """
    import random
    rng = random.Random(seed)

    if n >= len(samples):
        log.warning("n=%d >= pool size=%d — returning full pool", n, len(samples))
        return list(samples)

    by_label = defaultdict(list)
    for s in samples:
        by_label[s["label"]].append(s)

    labels = sorted(by_label.keys())
    n_labels = len(labels)
    
    # Initial quotas
    quotas = {l: n // n_labels for l in labels}
    for i in range(n % n_labels):
        quotas[labels[i]] += 1

    # Adjust quotas if some labels don't have enough items
    while True:
        shortfall = 0
        labels_with_extra = []
        for l in labels:
            if quotas[l] > len(by_label[l]):
                shortfall += quotas[l] - len(by_label[l])
                quotas[l] = len(by_label[l])
            elif quotas[l] < len(by_label[l]):
                labels_with_extra.append(l)
        
        if shortfall == 0 or not labels_with_extra:
            break
            
        # Distribute shortfall
        for i in range(shortfall):
            quotas[labels_with_extra[i % len(labels_with_extra)]] += 1

    selected = []
    for l in labels:
        selected.extend(rng.sample(by_label[l], quotas[l]))

    if len(selected) < n:
        log.warning("stratified_sample: Requested %d, but only found %d.", n, len(selected))

    rng.shuffle(selected)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset builder (main entry point for training scripts)
# ══════════════════════════════════════════════════════════════════════════════

def build_datasets(
    domain: str,
    setting: str,
    target_lang: str,
    processed_dir: str,
    n_shot: int | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Construct standardized training, validation, and test partitions for a cross-lingual ABSA experiment.

    Transfer Settings:
        - `s1` (Zero-shot): Train strictly on English (`en`), evaluate on `target_lang`.
        - `s2` (Few-shot): Train on English (`en`) + `n_shot` stratified target language samples.
        - `s3` (Full-target): Train on English (`en`) + full target language training dataset.

    Args:
        domain (str): Domain identifier (`"restaurant"` or `"phone"`).
        setting (str): Experimental transfer paradigm (`"s1"`, `"s2"`, or `"s3"`).
        target_lang (str): Target evaluation language code (`"vi"`, `"de"`, or `"zh"`).
        processed_dir (str): Path to root processed data directory.
        n_shot (Optional[int]): Number of few-shot target samples required when `setting="s2"`.
        seed (int): Random seed for stratified few-shot sampling. Defaults to 42.

    Returns:
        dict[str, list[dict]]: Dictionary containing `training`, `val`, `test_en`, and `test_target` splits.
    """
    # Load base splits
    en_train = load_split(domain, "en", "training", processed_dir)
    en_val   = load_split(domain, "en", "val",   processed_dir)
    en_test  = load_split(domain, "en", "test",  processed_dir)
    tgt_test = load_split(domain, target_lang, "test", processed_dir)

    if setting == "s1":
        train = en_train
        val   = en_val
        assert_no_leakage(train, ["en"], "s1")

    elif setting == "s2":
        if n_shot is None:
            raise ValueError("n_shot is required for setting 's2'")
        tgt_train_pool = load_split(domain, target_lang, "training", processed_dir)
        tgt_val        = load_split(domain, target_lang, "val",   processed_dir)
        sampled_tgt    = stratified_sample(tgt_train_pool, n_shot, seed)
        train = en_train + sampled_tgt
        val   = en_val + tgt_val
        assert_no_leakage(train, ["en", target_lang], f"s2_n{n_shot}")

    elif setting == "s3":
        tgt_train = load_split(domain, target_lang, "training", processed_dir)
        tgt_val   = load_split(domain, target_lang, "val",   processed_dir)
        train = en_train + tgt_train
        val   = en_val + tgt_val
        assert_no_leakage(train, ["en", target_lang], "s3")

    else:
        raise ValueError(f"Unknown setting '{setting}'. Use 's1', 's2', or 's3'.")

    # Final checks: test sets must never appear in training
    assert_test_not_seen(train, tgt_test)
    assert_test_not_seen(train, en_test)

    result = {
        "training": train,
        "val": val,
        "test_en": en_test,
        "test_target": tgt_test,
    }

    # Log statistics
    for key, samples in result.items():
        lang_dist = Counter(s["lang"] for s in samples)
        label_dist = Counter(s["sentiment"] for s in samples)
        impl_count = sum(1 for s in samples if s["is_implicit"])
        log.info(
            "[%s/%s/%s] %s | n=%d | langs=%s | labels=%s | implicit=%.1f%%",
            domain, setting, key,
            target_lang, len(samples),
            dict(lang_dist), dict(label_dist),
            impl_count / max(len(samples), 1) * 100,
        )

    return result
