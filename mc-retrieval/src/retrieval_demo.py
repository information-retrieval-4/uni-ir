"""Interactive Text→Build Retrieval Demo.

Usage:
  python src/retrieval_demo.py --config configs/pb_s1s2_semantic_init.yaml
  python src/retrieval_demo.py --config configs/pb_s1s2_semantic_init.yaml --top_k 3 --split test
"""

import argparse
import textwrap
import sys
import os

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import create_dataloaders
from model_pointbert import DualEncoderPointBERT
from utils import load_config, set_seed, get_device, load_checkpoint


@torch.no_grad()
def build_gallery(model, loader, device, dataset):
    """Pre-encode all voxels in the gallery + collect metadata from dataset."""
    model.eval()
    voxel_embs = []

    for _, voxels, _ in tqdm(loader, desc="Building gallery"):
        voxels = voxels.to(device)
        v_emb = model.encode_voxel(voxels)
        voxel_embs.append(v_emb.cpu())

    voxel_embs = torch.cat(voxel_embs, dim=0)

    titles = [txt[:120] if txt else "" for txt in dataset.texts]
    subtitles = dataset.categories
    descriptions = [""] * len(titles)   # not stored in dataset
    imgs = [""] * len(titles)           # not stored in dataset

    return voxel_embs, titles, subtitles, descriptions, imgs


def retrieve(query_text, model, device, gallery_embs, gallery_meta, top_k=5):
    """Encode query text → retrieve top-K voxels."""
    model.eval()
    query_emb = model.encode_text([query_text]).cpu()  # (1, D)
    sim = (query_emb @ gallery_embs.T).squeeze(0)       # (N,)
    top_k = min(top_k, len(sim))
    top_vals, top_idx = torch.topk(sim, top_k)

    titles, subtitles, descriptions, imgs = gallery_meta
    results = []
    for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
        results.append({
            "rank": len(results) + 1,
            "score": score,
            "title": titles[idx],
            "subtitle": subtitles[idx],
            "description": descriptions[idx],
            "img": imgs[idx],
        })
    return results


def display_results(query, results):
    """Pretty-print retrieval results."""
    print(f"\n{'='*72}")
    print(f"  Query: {query}")
    print(f"{'='*72}\n")

    for r in results:
        title = r["title"] if r["title"] else "(no title)"
        sub = r["subtitle"] if r["subtitle"] else "(no category)"

        print(f"  #{r['rank']}  sim={r['score']:.4f}")
        print(f"       {title}")
        print(f"       {sub}")
        if r.get("description"):
            desc = textwrap.shorten(r["description"], width=100, placeholder="...")
            print(f"       {desc}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--query", type=str, default=None,
                        help="Single query (non-interactive mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or f"{cfg['training']['checkpoint_dir']}/best.pt"
    ckpt = load_checkpoint(ckpt_path, device)
    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, block_mapping, num_blocks = \
        create_dataloaders(cfg)

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    gallery_loader = loader_map[args.split]
    print(f"Gallery split: {args.split} ({len(gallery_loader.dataset)} samples)")

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualEncoderPointBERT(cfg, num_block_types=num_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Build gallery ──────────────────────────────────────────────────────
    gallery_dataset = gallery_loader.dataset
    gallery_embs, titles, subtitles, descriptions, imgs = \
        build_gallery(model, gallery_loader, device, gallery_dataset)

    gallery_meta = (titles, subtitles, descriptions, imgs)
    print(f"Gallery: {len(gallery_embs)} voxel embeddings\n")

    # ── Run ────────────────────────────────────────────────────────────────
    if args.query:
        results = retrieve(args.query, model, device, gallery_embs, gallery_meta, args.top_k)
        display_results(args.query, results)
    else:
        print("Interactive retrieval mode. Type a query or 'q' to quit.\n")
        while True:
            try:
                query = input("  Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if query.lower() in ("q", "quit", "exit"):
                break
            if not query:
                continue

            results = retrieve(query, model, device, gallery_embs, gallery_meta, args.top_k)
            display_results(query, results)


if __name__ == "__main__":
    main()
