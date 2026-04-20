"""
WEMIR CBIR Demo
===============
This builds an index and allows for querying of images.

Usage:
    # Build index from image directory
    python demo.py build --data ./corel --output index.pkl

    # Query with an image
    python demo.py query --index index.pkl --image path/to/query.jpg --top_k 10

    # Evaluate precision/recall on random samples
    python demo.py evaluate --index index.pkl --data ./corel --samples 20
"""

import argparse
import sys
import random
from pathlib import Path

import cv2
import numpy as np

from wemir import WEMIRIndex


def cmd_build(args):
    """Build the WEMIR index from a directory of images."""
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Error: data directory '{data_dir}' not found")
        sys.exit(1)

    index = WEMIRIndex(svd_rank=args.svd_rank)
    index.build(data_dir)
    index.save(args.output)


def cmd_query(args):
    """Query the index with an image and display results."""
    index = WEMIRIndex.load(args.index)

    results = index.query(
        args.image,
        top_k=args.top_k,
        metric=args.metric,
    )

    query_label = Path(args.image).parent.name
    print(f"\nQuery: {args.image} (category: {query_label})")
    print(f"Metric: {args.metric}")
    print(f"{'Rank':<6} {'Distance':<14} {'Category':<20} {'File'}")
    print("-" * 80)

    for rank, (path, dist, label) in enumerate(results, 1):
        match = "✓" if label == query_label else "✗"
        print(f"{rank:<6} {dist:<14.4f} {label:<20} {Path(path).name} {match}")

    # Show results visually
    if not args.no_display:
        _display_results(args.image, results, args.output_image)


def cmd_evaluate(args):
    """Evaluate precision/recall on random query samples."""
    index = WEMIRIndex.load(args.index)

    # Pick random images from the index as queries
    all_paths = list(index.features.keys())
    n_samples = min(args.samples, len(all_paths))
    sample_paths = random.sample(all_paths, n_samples)

    precisions = []
    recalls = []
    per_category = {}

    for qpath in sample_paths:
        result = index.evaluate(qpath, top_k=args.top_k, metric=args.metric)
        precisions.append(result["precision"])
        recalls.append(result["recall"])

        cat = result["query_label"]
        if cat not in per_category:
            per_category[cat] = {"precisions": [], "recalls": []}
        per_category[cat]["precisions"].append(result["precision"])
        per_category[cat]["recalls"].append(result["recall"])

    # Print per-category results
    print(f"\n{'Category':<20} {'Avg Precision':<16} {'Avg Recall':<16} {'Samples'}")
    print("-" * 68)

    for cat in sorted(per_category.keys()):
        data = per_category[cat]
        avg_p = np.mean(data["precisions"])
        avg_r = np.mean(data["recalls"])
        n = len(data["precisions"])
        print(f"{cat:<20} {avg_p:<16.4f} {avg_r:<16.4f} {n}")

    print("-" * 68)
    print(
        f"{'AVERAGE':<20} {np.mean(precisions):<16.4f} {np.mean(recalls):<16.4f} {n_samples}"
    )


def _display_results(query_path, results, output_path=None):
    """Create a visual grid of query + results and display/save it."""
    query_img = cv2.imread(str(query_path))
    if query_img is None:
        return

    # Resize all to a uniform size for the grid
    thumb_size = (150, 150)
    query_thumb = cv2.resize(query_img, thumb_size)

    # Add green border to query
    cv2.rectangle(query_thumb, (0, 0), (149, 149), (0, 200, 0), 3)

    result_thumbs = []
    query_label = Path(query_path).parent.name

    for path, dist, label in results:
        img = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, thumb_size)
        # Green border if same category, red if different
        color = (0, 200, 0) if label == query_label else (0, 0, 200)
        cv2.rectangle(thumb, (0, 0), (149, 149), color, 2)
        result_thumbs.append(thumb)

    if not result_thumbs:
        return

    # Build grid: query on top, results below in rows of 5
    cols = 5
    rows_needed = (len(result_thumbs) + cols - 1) // cols

    # Query row
    query_row = np.zeros((thumb_size[1] + 30, thumb_size[0] * cols, 3), dtype=np.uint8)
    query_row[30 : 30 + thumb_size[1], 0 : thumb_size[0]] = query_thumb
    cv2.putText(
        query_row, "QUERY", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
    )

    # Result rows
    result_grid = np.zeros(
        (rows_needed * (thumb_size[1] + 30), thumb_size[0] * cols, 3), dtype=np.uint8
    )
    for idx, thumb in enumerate(result_thumbs):
        r = idx // cols
        c = idx % cols
        y = r * (thumb_size[1] + 30) + 30
        x = c * thumb_size[0]
        result_grid[y : y + thumb_size[1], x : x + thumb_size[0]] = thumb

        # Rank label
        rank_text = f"#{idx + 1}"
        cv2.putText(
            result_grid,
            rank_text,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Stack vertically
    canvas = np.vstack([query_row, result_grid])

    if output_path:
        cv2.imwrite(str(output_path), canvas)
        print(f"\nResults saved to {output_path}")
    else:
        cv2.imshow("WEMIR Results", canvas)
        print("\nPress any key to close the results window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="WEMIR - Weighted Edge Matching Information Retrieval"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build
    build_parser = subparsers.add_parser("build", help="Build feature index")
    build_parser.add_argument("--data", required=True, help="Path to image directory")
    build_parser.add_argument("--output", default="index.pkl", help="Output index file")
    build_parser.add_argument(
        "--svd_rank", type=int, default=None, help="SVD rank (None = auto)"
    )

    # Query
    query_parser = subparsers.add_parser("query", help="Query with an image")
    query_parser.add_argument("--index", required=True, help="Path to index file")
    query_parser.add_argument("--image", required=True, help="Query image path")
    query_parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    query_parser.add_argument(
        "--metric", default="euclidean", choices=["euclidean", "manhattan"]
    )
    query_parser.add_argument(
        "--no_display", action="store_true", help="Skip visual display"
    )
    query_parser.add_argument(
        "--output_image",
        default=None,
        help="Save result grid to file instead of displaying",
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate precision/recall")
    eval_parser.add_argument("--index", required=True, help="Path to index file")
    eval_parser.add_argument(
        "--data", default=None, help="Image directory (for finding categories)"
    )
    eval_parser.add_argument(
        "--samples", type=int, default=20, help="Number of random query samples"
    )
    eval_parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    eval_parser.add_argument(
        "--metric", default="euclidean", choices=["euclidean", "manhattan"]
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
