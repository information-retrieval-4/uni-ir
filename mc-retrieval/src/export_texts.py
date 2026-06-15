"""Export text inputs (what enters the model) + metadata to CSV for review.

Usage:
  python src/export_texts.py --config configs/pb_s1s2_semantic_init.yaml
  python src/export_texts.py --config configs/pb_s1s2_semantic_init.yaml --split all --max_rows 50
"""

import argparse
import sys
import os
import csv

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import create_dataloaders, build_text, clean_text
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="text_review.csv")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--max_rows", type=int, default=0,
                        help="Max rows to export (0 = all)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load dataloaders (builds block mapping but we just need the texts)
    train_loader, val_loader, test_loader, block_mapping, num_blocks = \
        create_dataloaders(cfg)

    # Get datasets
    datasets = {}
    if args.split == "all":
        datasets = {
            "train": train_loader.dataset,
            "val": val_loader.dataset,
            "test": test_loader.dataset,
        }
    else:
        loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
        datasets = {args.split: loader_map[args.split].dataset}

    rows = []
    total = 0

    for split_name, ds in datasets.items():
        for i in tqdm(range(len(ds)), desc=f"Exporting {split_name}"):
            if args.max_rows and total >= args.max_rows:
                break

            text = ds.texts[i]
            category = ds.categories[i]

            # Extract original fields if still in text format
            # Try to recover title from first part before category
            title = ""
            for cat_name in [
                "Land Structure Map", "3D Art Map", "Redstone Device Map",
                "Other Map", "Air Structure Map", "Complex Map",
                "Pixel Art Map", "Piston Map", "Water Structure Map",
                "Environment / Landscaping Map", "Challenge / Adventure Map",
                "Minecart Map", "Underground Structure Map",
                "Nether Structure Map", "Music Map", "Educational Map",
            ]:
                idx = text.find(cat_name)
                if idx > 0:
                    title = text[:idx].strip()
                    break

            rows.append({
                "split": split_name,
                "category": category,
                "title_extracted": title[:120],
                "text_input_len": len(text),
                "text_input": text,
            })
            total += 1

    # Write CSV
    fieldnames = ["split", "category", "title_extracted", "text_input_len", "text_input"]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExported {len(rows)} rows to '{args.output}'")


if __name__ == "__main__":
    main()
