"""Visualize the shared text-voxel embedding space using UMAP."""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder
from utils import load_config, set_seed, get_device, load_checkpoint


# register a nicer font if available
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
})


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    text_embs, voxel_embs, categories = [], [], []
    for texts, voxels, cats in tqdm(loader, desc="Extracting embeddings", leave=False):
        voxels = voxels.to(device)
        t_emb, v_emb = model(texts, voxels)
        text_embs.append(t_emb.cpu())
        voxel_embs.append(v_emb.cpu())
        categories.extend(cats)
    return (torch.cat(text_embs, 0).numpy(),
            torch.cat(voxel_embs, 0).numpy(),
            np.array(categories))


def plot_embedding_space(text_embs, voxel_embs, categories, save_path="embedding_space.png"):
    all_embs = np.concatenate([text_embs, voxel_embs], axis=0)
    N = len(text_embs)

    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(all_embs)
    text_coords = coords[:N]
    voxel_coords = coords[N:]

    # top categories
    unique_cats, counts = np.unique(categories, return_counts=True)
    top_k = 7
    top_cats = unique_cats[np.argsort(-counts)][:top_k]

    palette = [
        "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
        "#76b7b2", "#edc948", "#b07aa1", "#9c755f",
    ]
    cat_colors = {cat: palette[i] for i, cat in enumerate(top_cats)}

    short_names = {
        "Land Structure Map": "Land Structure",
        "3D Art Map": "3D Art",
        "Redstone Device Map": "Redstone",
        "Other Map": "Other",
        "Air Structure Map": "Air Structure",
        "Complex Map": "Complex",
        "Pixel Art Map": "Pixel Art",
        "Water Structure Map": "Water Structure",
        "Environment / Landscaping Map": "Environment",
        "Piston Map": "Piston",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#fafafa")
    for ax in axes:
        ax.set_facecolor("#fafafa")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Panel 1: by category ---
    ax = axes[0]
    ax.set_title("Shared Embedding Space — by Category")

    # plot "other" first (background)
    other_mask = ~np.isin(categories, top_cats)
    if other_mask.any():
        other_all = np.concatenate([other_mask, other_mask])
        ax.scatter(coords[other_all, 0], coords[other_all, 1],
                   c="#d0d0d0", s=6, alpha=0.2, zorder=1)

    # plot each category (text + voxel together)
    for cat in top_cats:
        cat_mask = categories == cat
        cat_all = np.concatenate([cat_mask, cat_mask])
        ax.scatter(coords[cat_all, 0], coords[cat_all, 1],
                   c=cat_colors[cat], s=14, alpha=0.55, zorder=2,
                   label=short_names.get(cat, cat), edgecolors="white",
                   linewidths=0.2)

    ax.legend(fontsize=9, markerscale=2.5, loc="lower left",
              framealpha=0.9, edgecolor="#ccc", fancybox=True)

    # --- Panel 2: by modality with paired lines ---
    ax = axes[1]
    ax.set_title("Cross-Modal Alignment — Text vs Voxel")

    # all voxels
    ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1],
               c="#4e79a7", s=12, alpha=0.35, zorder=2, label="Voxel")
    # all text
    ax.scatter(text_coords[:, 0], text_coords[:, 1],
               c="#e15759", s=12, alpha=0.35, marker="x",
               linewidths=0.8, zorder=2, label="Text")

    # draw paired lines for a subset
    np.random.seed(42)
    sample_idx = np.random.choice(N, min(60, N), replace=False)
    for idx in sample_idx:
        cat = categories[idx]
        color = cat_colors.get(cat, "#999999")
        tx, ty = text_coords[idx]
        vx, vy = voxel_coords[idx]
        ax.plot([tx, vx], [ty, vy], c=color, alpha=0.25, linewidth=0.6, zorder=1)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7",
               markersize=9, label="Voxel embedding"),
        Line2D([0], [0], marker="x", color="w", markeredgecolor="#e15759",
               markersize=9, markeredgewidth=2, label="Text embedding"),
        Line2D([0], [0], color="#999", alpha=0.5, linewidth=1.5,
               label="Paired (same item)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower left",
              framealpha=0.9, edgecolor="#ccc", fancybox=True)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="#fafafa")
    print(f"Saved to {save_path}")
    plt.close()

    return coords, top_cats, cat_colors, short_names


def plot_per_category(coords, categories, top_cats, cat_colors, short_names,
                      save_path="embedding_per_category.png"):
    """Grid of small multiples: one category highlighted per subplot."""
    all_cats = np.concatenate([categories, categories])
    n_cats = len(top_cats)
    cols = 4
    rows = (n_cats + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), facecolor="white")
    axes = axes.flatten()

    for i, cat in enumerate(top_cats):
        ax = axes[i]
        ax.set_facecolor("white")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#ddd")

        # gray background (all points)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="#e8e8e8", s=6, alpha=0.3, zorder=1)

        # highlight this category
        cat_mask = all_cats == cat
        count = cat_mask.sum() // 2  # each item appears twice (text + voxel)
        ax.scatter(coords[cat_mask, 0], coords[cat_mask, 1],
                   c=cat_colors[cat], s=22, alpha=0.7, zorder=2,
                   edgecolors="white", linewidths=0.3)

        name = short_names.get(cat, cat)
        ax.set_title(f"{name}  (n={count})", fontsize=13, fontweight="bold",
                     color=cat_colors[cat])

    # hide unused axes
    for j in range(n_cats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-Category Embedding Clusters", fontsize=18,
                 fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding space")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="embedding_space.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    _, _, test_loader, block_mapping, num_blocks, _ = create_dataloaders(cfg)
    ckpt = load_checkpoint(args.checkpoint, device)
    model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])

    text_embs, voxel_embs, categories = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(text_embs)} embeddings (dim={text_embs.shape[1]})")

    coords, top_cats, cat_colors, short_names = \
        plot_embedding_space(text_embs, voxel_embs, categories, args.output)

    # per-category highlight grid
    cat_path = args.output.replace(".png", "_per_category.png")
    plot_per_category(coords, categories, top_cats, cat_colors, short_names, cat_path)


if __name__ == "__main__":
    main()

