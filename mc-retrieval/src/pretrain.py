"""Masked Voxel Modeling (MVM) pretraining for the voxel encoder.

Masks ~20% of non-air blocks and trains a U-Net-style encoder-decoder
to reconstruct them. The encoder shares architecture with VoxelEncoder
so weights transfer directly.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import build_block_mapping, remap_voxel
from utils import load_config, set_seed, get_device, save_checkpoint


# ---------------------------------------------------------------------------
# Dataset (voxel-only, no text needed)
# ---------------------------------------------------------------------------

class VoxelOnlyDataset(Dataset):
    """Dataset that returns only voxel grids for self-supervised pretraining."""

    def __init__(self, df: pd.DataFrame, block_mapping: dict):
        self.voxels = df["voxel_data"].tolist()
        self.block_mapping = block_mapping

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        return remap_voxel(self.voxels[idx], self.block_mapping)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def create_mask(voxels: torch.LongTensor, mask_ratio: float = 0.2):
    """Randomly mask non-air blocks.

    Args:
        voxels: (B, 32, 32, 32) block IDs
        mask_ratio: fraction of non-air blocks to mask

    Returns:
        masked_voxels: (B, 32, 32, 32) with masked positions set to mask_token_id
        mask: (B, 32, 32, 32) bool tensor indicating masked positions
    """
    non_air = voxels != 0                                       # (B, 32, 32, 32)
    rand = torch.rand_like(voxels, dtype=torch.float32)
    mask = non_air & (rand < mask_ratio)
    return mask


# ---------------------------------------------------------------------------
# Augmentation (same as training)
# ---------------------------------------------------------------------------

def augment_voxels(voxels: torch.LongTensor) -> torch.LongTensor:
    """Random 90° Y-rotation + horizontal flips."""
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        voxels = torch.rot90(voxels, k, dims=(1, 3))
    if torch.rand(1).item() > 0.5:
        voxels = voxels.flip(dims=(1,))
    if torch.rand(1).item() > 0.5:
        voxels = voxels.flip(dims=(3,))
    return voxels


# ---------------------------------------------------------------------------
# Model: U-Net style encoder-decoder
# ---------------------------------------------------------------------------

class MaskedVoxelModel(nn.Module):
    """U-Net encoder-decoder for masked block prediction.

    Encoder architecture matches VoxelEncoder exactly so weights
    can be transferred after pretraining.
    """

    def __init__(
        self,
        num_block_types: int = 256,
        block_embed_dim: int = 32,
        channels: list[int] = [64, 128, 256],
        dropout: float = 0.3,
        mask_ratio: float = 0.2,
    ):
        super().__init__()
        self.num_block_types = num_block_types
        self.mask_token_id = num_block_types   # extra token for [MASK]
        self.mask_ratio = mask_ratio

        # +1 for mask token
        self.block_embedding = nn.Embedding(num_block_types + 1, block_embed_dim)

        # --- Encoder (mirrors VoxelEncoder.conv_stack) ---
        # Block 1: 32³ → 16³
        self.enc1 = nn.Sequential(
            nn.Conv3d(block_embed_dim, channels[0], 3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool1 = nn.MaxPool3d(2)

        # Block 2: 16³ → 8³
        self.enc2 = nn.Sequential(
            nn.Conv3d(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck: 8³ (no pooling)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

        # --- Decoder ---
        # Up 2: 8³ → 16³, concat with enc2
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(channels[1] * 2, channels[1], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
        )

        # Up 1: 16³ → 32³, concat with enc1
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(channels[0] * 2, channels[0], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
        )

        # Prediction head: per-voxel block classification
        self.pred_head = nn.Conv3d(channels[0], num_block_types, 1)

    def forward(self, voxels: torch.LongTensor):
        """
        Args:
            voxels: (B, 32, 32, 32) original block IDs
        Returns:
            logits: (B, num_blocks, 32, 32, 32) predictions
            mask:   (B, 32, 32, 32) bool mask of what was masked
        """
        # Create mask and apply
        mask = create_mask(voxels, self.mask_ratio)
        masked_voxels = voxels.clone()
        masked_voxels[mask] = self.mask_token_id

        # Embed
        x = self.block_embedding(masked_voxels)          # (B, 32, 32, 32, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()        # (B, D, 32, 32, 32)

        # Encoder
        e1 = self.enc1(x)                                  # (B, C0, 32, 32, 32)
        e2 = self.enc2(self.pool1(e1))                     # (B, C1, 16, 16, 16)
        bn = self.bottleneck(self.pool2(e2))               # (B, C2, 8, 8, 8)

        # Decoder with skip connections
        d2 = self.up2(bn)                                  # (B, C1, 16, 16, 16)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))         # (B, C1, 16, 16, 16)

        d1 = self.up1(d2)                                  # (B, C0, 32, 32, 32)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))         # (B, C0, 32, 32, 32)

        # Predict
        logits = self.pred_head(d1)                        # (B, num_blocks, 32, 32, 32)

        return logits, mask

    def get_encoder_state_dict(self):
        """Extract encoder weights in VoxelEncoder-compatible format.

        Maps our named encoder blocks to VoxelEncoder.conv_stack indices:
            enc1.{0,1}     → conv_stack.{0,1}       (Conv3d, BN)
            enc2.{0,1}     → conv_stack.{5,6}        (Conv3d, BN)
            bottleneck.{0,1} → conv_stack.{10,11}    (Conv3d, BN)
        Block embedding is copied directly (without the extra mask token).
        """
        state = {}

        # Block embedding (drop mask token)
        state["block_embedding.weight"] = \
            self.block_embedding.weight[:self.num_block_types].clone()

        # Map encoder blocks → conv_stack indices
        # enc1 → conv_stack.0-3 (Conv, BN, GELU=no params, Dropout=no params)
        # In Sequential, Conv is .0, BN is .1
        mapping = {
            "enc1": 0,    # conv_stack indices 0,1
            "enc2": 5,    # conv_stack indices 5,6
            "bottleneck": 10,  # conv_stack indices 10,11
        }

        for block_name, stack_offset in mapping.items():
            block = getattr(self, block_name)
            # Conv3d (index 0 in the block)
            conv = block[0]
            state[f"conv_stack.{stack_offset}.weight"] = conv.weight.clone()
            state[f"conv_stack.{stack_offset}.bias"] = conv.bias.clone()
            # BatchNorm3d (index 1 in the block)
            bn = block[1]
            state[f"conv_stack.{stack_offset + 1}.weight"] = bn.weight.clone()
            state[f"conv_stack.{stack_offset + 1}.bias"] = bn.bias.clone()
            state[f"conv_stack.{stack_offset + 1}.running_mean"] = bn.running_mean.clone()
            state[f"conv_stack.{stack_offset + 1}.running_var"] = bn.running_var.clone()
            state[f"conv_stack.{stack_offset + 1}.num_batches_tracked"] = bn.num_batches_tracked.clone()

        return state


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pretrain(cfg: dict):
    """Run masked voxel modeling pretraining."""
    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    pt_cfg = cfg.get("pretraining", {})
    mask_ratio = pt_cfg.get("mask_ratio", 0.2)
    epochs = pt_cfg.get("epochs", 200)
    batch_size = pt_cfg.get("batch_size", 256)
    lr = pt_cfg.get("lr", 1e-3)
    ckpt_dir = pt_cfg.get("checkpoint_dir", "checkpoints")

    # --- data (use ALL samples, no splits needed) ---
    df = pd.read_parquet(cfg["data"]["parquet_path"])
    print(f"Loaded {len(df)} samples for pretraining")

    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )
    num_blocks = cfg["data"]["max_block_types"]

    dataset = VoxelOnlyDataset(df, block_mapping)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # --- model ---
    model_cfg = cfg["model"]
    model = MaskedVoxelModel(
        num_block_types=num_blocks,
        block_embed_dim=model_cfg["block_embed_dim"],
        channels=model_cfg["voxel_channels"],
        dropout=model_cfg.get("dropout", 0.3),
        mask_ratio=mask_ratio,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"MVM model parameters: {param_count:,}")

    # --- optimizer ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- training ---
    print(f"\nStarting MVM pretraining for {epochs} epochs...")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Batch size: {batch_size}")
    print()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"  epoch {epoch:3d}", leave=False)
        for voxels in pbar:
            voxels = voxels.to(device)
            voxels = augment_voxels(voxels)

            logits, mask = model(voxels)

            # Loss only on masked positions
            # logits: (B, C, 32, 32, 32), targets: (B, 32, 32, 32)
            loss = F.cross_entropy(logits, voxels, reduction="none")  # (B, 32, 32, 32)
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accuracy on masked positions
            preds = logits.argmax(dim=1)  # (B, 32, 32, 32)
            correct = ((preds == voxels) & mask).sum().item()
            n_masked = mask.sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_masked += n_masked
            num_batches += 1

            acc = correct / max(n_masked, 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_correct / max(total_masked, 1)
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  mask_acc={avg_acc:.4f}  "
            f"lr={lr_current:.2e}"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "encoder_state": model.get_encoder_state_dict(),
                    "loss": avg_loss,
                    "accuracy": avg_acc,
                    "cfg": cfg,
                },
                os.path.join(ckpt_dir, "pretrained_voxel.pt"),
            )

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain voxel encoder via masked voxel modeling")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pretrain(cfg)
