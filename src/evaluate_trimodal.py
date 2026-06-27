"""Trimodal retrieval evaluation: Text↔Voxel, Text↔Image, Image↔Voxel."""

import argparse

import torch
import numpy as np
from tqdm import tqdm

from dataset_trimodal import create_trimodal_dataloaders
from model_trimodal import TriModalEncoder
from evaluate import compute_retrieval_metrics, compute_category_metrics
from utils import load_config, set_seed, get_device, load_checkpoint


@torch.no_grad()
def extract_trimodal_embeddings(model, loader, device):
    """Encode all batches → (text_embs, image_embs, voxel_embs, categories)."""
    model.eval()
    text_embs, image_embs, voxel_embs = [], [], []
    all_categories = []

    for texts, images, voxels, categories in tqdm(loader, desc="Encoding", leave=False):
        images = images.to(device)
        voxels = voxels.to(device)
        t = model.encode_text(texts)
        i = model.encode_image(images)
        v = model.encode_voxel(voxels)
        text_embs.append(t.cpu())
        image_embs.append(i.cpu())
        voxel_embs.append(v.cpu())
        all_categories.extend(categories)

    return (
        torch.cat(text_embs, 0),
        torch.cat(image_embs, 0),
        torch.cat(voxel_embs, 0),
        all_categories,
    )


def print_metrics(label: str, metrics: dict):
    print(f"\n  {label}:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")


def evaluate_trimodal(cfg: dict, checkpoint_path: str = None):
    """Full trimodal retrieval evaluation on the test set."""
    set_seed(cfg["data"]["seed"])
    device = get_device()

    if checkpoint_path is None:
        checkpoint_path = f"{cfg['training']['checkpoint_dir']}/best.pt"
    ckpt = load_checkpoint(checkpoint_path, device)

    _, _, test_loader, block_mapping, num_blocks, processor = create_trimodal_dataloaders(cfg)

    model = TriModalEncoder(cfg, num_block_types=num_blocks, processor=processor).to(device)
    model.load_state_dict(ckpt["model_state"])

    text_embs, image_embs, voxel_embs, categories = extract_trimodal_embeddings(
        model, test_loader, device
    )
    N = len(categories)
    print(f"\nExtracted {N} embeddings (dim={text_embs.shape[1]})")

    unique_cats, cat_counts = np.unique(categories, return_counts=True)
    print(f"Categories in test set: {len(unique_cats)}")
    for cat, cnt in sorted(zip(unique_cats, cat_counts), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {cnt}")

    ks = cfg["eval"]["recall_k"]

    print("\n" + "=" * 55)
    print("INSTANCE-LEVEL RETRIEVAL")
    print("=" * 55)

    # NOTE: no scores[i] -= 1e9 — these are cross-modal pairs, not self-retrieval
    print_metrics("Text → Voxel", compute_retrieval_metrics(text_embs, voxel_embs, ks=ks))
    print_metrics("Voxel → Text", compute_retrieval_metrics(voxel_embs, text_embs, ks=ks))
    print_metrics("Text → Image", compute_retrieval_metrics(text_embs, image_embs, ks=ks))
    print_metrics("Image → Voxel", compute_retrieval_metrics(image_embs, voxel_embs, ks=ks))

    print("\n" + "=" * 55)
    print("CATEGORY-LEVEL RETRIEVAL")
    print("=" * 55)

    cats = categories
    print_metrics("Text → Voxel (cat)", compute_category_metrics(text_embs, voxel_embs, cats, cats, ks=ks))
    print_metrics("Voxel → Text (cat)", compute_category_metrics(voxel_embs, text_embs, cats, cats, ks=ks))
    print_metrics("Text → Image (cat)", compute_category_metrics(text_embs, image_embs, cats, cats, ks=ks))
    print_metrics("Image → Voxel (cat)", compute_category_metrics(image_embs, voxel_embs, cats, cats, ks=ks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trimodal retrieval model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_trimodal(cfg, args.checkpoint)
