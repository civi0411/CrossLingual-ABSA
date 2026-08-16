"""
Download pretrained multilingual transformer backbones (`mBERT`, `XLM-R`, `mT5-small`) from HuggingFace and cache them locally.
Run this setup script once prior to initiating offline training experiments.

Usage:
    python scripts/download_models.py --output_dir ./pretrained_models
"""

import argparse
from pathlib import Path
from transformers import (
    BertTokenizer,
    BertModel,
    XLMRobertaTokenizer,
    XLMRobertaModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
)


def download_mbert(save_dir):
    """Download and cache `bert-base-multilingual-cased` weights and tokenizer locally."""
    model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    save_path = Path(save_dir) / "mbert"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Downloaded mBERT to {save_path}")


def download_xlmr(save_dir):
    """Download and cache `xlm-roberta-base` weights and tokenizer locally."""
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name)
    save_path = Path(save_dir) / "xlmr"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Downloaded XLM-R to {save_path}")


def download_mt5(save_dir):
    """Download and cache `google/mt5-small` weights and tokenizer locally."""
    model_name = "google/mt5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    save_path = Path(save_dir) / "mt5"
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Downloaded mT5 to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download multilingual models for ABSA project"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pretrained_models",
        help="Directory to save downloaded models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mbert", "xlmr", "mt5"],
        default=["mbert", "xlmr", "mt5"],
        help="Which models to download (default: all)",
    )
    args = parser.parse_args()

    print("Starting download...")
    if "mbert" in args.models:
        download_mbert(args.output_dir)
    if "xlmr" in args.models:
        download_xlmr(args.output_dir)
    if "mt5" in args.models:
        download_mt5(args.output_dir)
    print("All downloads completed.")


if __name__ == "__main__":
    main()