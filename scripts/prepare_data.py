#!/usr/bin/env python
"""
Prepare raw datasets for the cross-lingual ABSA pipeline:
    1. Ingestion: Download/parse M-ABSA XML data and convert to standardized JSONL files (`domain/lang/split`).
    2. Validation: Perform integrity checks across splits.

Usage:
    python scripts/prepare_data.py --domains restaurant phone
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from src.data.ingest import load_domain
from src.utils.common import ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["restaurant", "phone"])
    parser.add_argument("--processed_dir", default="data/processed")
    args = parser.parse_args()

    for domain in args.domains:
        log.info(f"Processing domain: {domain}")
        result = load_domain(domain, args.processed_dir)
        log.info(f"Ingested {len(result)} language-split files for {domain}")

    log.info("Data preparation complete. You can now run experiments.")


if __name__ == "__main__":
    main()
