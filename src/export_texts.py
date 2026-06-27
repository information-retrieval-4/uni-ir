"""Export text inputs (what enters the model) + metadata to CSV for review.

Usage:
  python src/export_texts.py --parquet data/data_with_voxel_names_multiview_image.parquet
  python src/export_texts.py --parquet data/data_with_voxel_names_multiview_image.parquet --max_rows 50
  python src/export_texts.py --config configs/pointbert/pb_s1s2_semantic_init.yaml
"""

import argparse
import sys
import os
import csv

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_text
from utils import load_config


def export_from_df(df, output, max_rows=0):
    rows = []
    for i in tqdm(range(len(df)), desc="Exporting"):
        if max_rows and i >= max_rows:
            break
        row = df.iloc[i]
        text = build_text(row)
        rows.append({
            "index": i,
            "title": str(row.get("title", "")),
            "subtitle": str(row.get("subtitle", "")),
            "description": str(row.get("description", ""))[:500],
            "tags": str(row.get("tags", "")),
            "text_input": text,
        })

    fieldnames = ["index", "title", "subtitle", "description", "tags", "text_input"]
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to '{output}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=None,
                        help="Direct parquet path (no config needed)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML path (reads parquet_path from config)")
    parser.add_argument("--output", type=str, default="text_review.csv")
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    if args.parquet:
        path = args.parquet
    elif args.config:
        cfg = load_config(args.config)
        path = cfg["data"]["parquet_path"]
    else:
        parser.error("Either --parquet or --config is required")

    df = pd.read_parquet(path, columns=["title", "subtitle", "description", "tags"])
    print(f"Loaded {len(df)} rows from {path}")
    export_from_df(df, args.output, args.max_rows)


if __name__ == "__main__":
    main()
