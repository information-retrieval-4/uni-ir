"""Visualize MVM reconstruction with 3D voxel rendering.

Layout: rows = mask ratios, columns = Original | Masked | Reconstructed
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from pretrain import MaskedVoxelModel, VoxelOnlyDataset, create_mask
from dataset import build_block_mapping
from utils import load_config, set_seed, get_device


def make_block_palette(n_blocks=256):
    """Generate a more visually distinct palette using HSL with better spread."""
    colors = {}
    colors[0] = (0, 0, 0, 0)  # air = fully transparent

    # use a curated set of base hues, then vary lightness
    base_hues = [0, 30, 60, 120, 180, 210, 270, 330]
    for i in range(1, n_blocks):
        hue = base_hues[i % len(base_hues)] + (i // len(base_hues)) * 7
        hue = hue % 360
        sat = 50 + (i * 13 % 40)
        light = 35 + (i * 17 % 45)
        # HSL to RGB via matplotlib
        rgb = mcolors.hsv_to_rgb([hue / 360, sat / 100, light / 100])
        colors[i] = (*rgb, 1.0)
    return colors


def vol_to_facecolors(vol, palette):
    """Convert block ID volume to RGBA facecolors for ax.voxels()."""
    rgba = np.zeros((*vol.shape, 4))
    for bid in np.unique(vol):
        if bid == 0:
            continue
        mask = vol == bid
        rgba[mask] = palette.get(int(bid), (0.5, 0.5, 0.5, 1.0))
    return rgba


def render_voxel(ax, vol, palette, title="", elev=30, azim=135):
    """Render voxel grid on a 3D axis. Only renders non-air blocks."""
    filled = vol != 0
    if not filled.any():
        ax.set_title(title, fontsize=9)
        return

    facecolors = vol_to_facecolors(vol, palette)

    ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.05)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=1)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.95, 0.95, 0.95, 0.3))
    ax.yaxis.pane.set_edgecolor((0.95, 0.95, 0.95, 0.3))
    ax.zaxis.pane.set_edgecolor((0.95, 0.95, 0.95, 0.3))
    ax.grid(False)


@torch.no_grad()
def get_masked_and_recon(model, voxel, mask_ratio, device):
    """Return the masked voxel (with holes) and reconstruction."""
    v = voxel.unsqueeze(0).to(device)
    original = v[0].cpu().numpy()

    mask = create_mask(v, mask_ratio=mask_ratio)
    m = mask[0].cpu().numpy()

    # masked version: original with masked positions set to air (holes)
    masked_display = original.copy()
    masked_display[m] = 0  # show holes as empty

    # reconstruction
    masked_input = v.clone()
    masked_input[mask] = model.mask_token_id

    x = model.block_embedding(masked_input)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool1(e1))
    bn = model.bottleneck(model.pool2(e2))
    d2 = model.up2(bn)
    d2 = model.dec2(torch.cat([d2, e2], dim=1))
    d1 = model.up1(d2)
    d1 = model.dec1(torch.cat([d1, e1], dim=1))
    logits = model.pred_head(d1)
    recon = logits.argmax(dim=1)[0].cpu().numpy()

    # reconstruction: keep original at non-masked, model prediction at masked
    recon_display = original.copy()
    recon_display[m] = recon[m]  # only replace masked positions

    acc = (recon[m] == original[m]).mean() if m.sum() > 0 else 1.0
    return masked_display, recon_display, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_voxel.pt")
    parser.add_argument("--output", type=str, default="mvm_reconstruction_3d.png")
    parser.add_argument("--sample-idx", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    df = pd.read_parquet(cfg["data"]["parquet_path"])
    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )
    test_df = df.iloc[int(len(df) * 0.9):]
    dataset = VoxelOnlyDataset(test_df, block_mapping)

    # pick a sample with moderate density (not too sparse, not too dense)
    if args.sample_idx is not None:
        sample = dataset[args.sample_idx]
        density = (sample != 0).float().mean().item()
    else:
        best_idx, best_score = 0, float("inf")
        target_density = 0.30  # aim for ~30% density
        for i in range(min(300, len(dataset))):
            v = dataset[i]
            d = (v != 0).float().mean().item()
            score = abs(d - target_density)
            if score < best_score:
                best_score = score
                best_idx = i
        sample = dataset[best_idx]
        density = (sample != 0).float().mean().item()
        print(f"Auto-picked sample {best_idx} (density={density:.1%})")

    # load model
    model_cfg = cfg["model"]
    model = MaskedVoxelModel(
        num_block_types=cfg["data"]["max_block_types"],
        block_embed_dim=model_cfg["block_embed_dim"],
        channels=model_cfg["voxel_channels"],
        dropout=0.0,
        mask_ratio=0.2,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    palette = make_block_palette(256)
    mask_ratios = [0.2, 0.5, 0.7, 0.9]

    # layout: rows = mask ratios, cols = Original | Masked | Reconstructed
    n_rows = len(mask_ratios)
    n_cols = 3
    fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), facecolor="white")

    original = sample.numpy()

    for row, ratio in enumerate(mask_ratios):
        masked_display, recon, acc = get_masked_and_recon(
            model, sample, ratio, device
        )

        # col 0: original
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 1, projection="3d")
        render_voxel(ax, original, palette,
                     title="Original" if row == 0 else "")
        if row == 0:
            ax.text2D(0.02, 0.95, "Original", transform=ax.transAxes,
                      fontsize=11, fontweight="bold", va="top")

        # col 1: masked (with holes)
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 2, projection="3d")
        render_voxel(ax, masked_display, palette,
                     title=f"Masked ({ratio:.0%})")

        # col 2: reconstructed
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + 3, projection="3d")
        render_voxel(ax, recon, palette,
                     title=f"Reconstructed (acc={acc:.1%})")

    fig.suptitle("Masked Voxel Modeling — Reconstruction",
                 fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout(pad=1.5)
    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
