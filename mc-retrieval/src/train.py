"""Training loop for the dual encoder model."""

import os
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder
from losses import CLIPLoss
from utils import load_config, set_seed, get_device, save_checkpoint


# ---------------------------------------------------------------------------
# Voxel augmentation
# ---------------------------------------------------------------------------

def augment_voxels(voxels: torch.LongTensor) -> torch.LongTensor:
    """Apply random 90° Y-axis rotation and horizontal flip to voxel grids.

    Args:
        voxels: (B, 32, 32, 32)
    Returns:
        augmented voxels, same shape
    """
    # random 90° rotations around Y axis (rotate in X-Z plane)
    # tensor is (B, X, Y, Z) — X=dim1, Y=dim2, Z=dim3
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        voxels = torch.rot90(voxels, k, dims=(1, 3))

    # random horizontal flip along X
    if torch.rand(1).item() > 0.5:
        voxels = voxels.flip(dims=(1,))

    # random horizontal flip along Z
    if torch.rand(1).item() > 0.5:
        voxels = voxels.flip(dims=(3,))

    return voxels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, augment=True):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for texts, voxels, _categories in pbar:
        voxels = voxels.to(device)

        if augment:
            voxels = augment_voxels(voxels)

        text_emb, voxel_emb = model(texts, voxels)
        loss = criterion(text_emb, voxel_emb)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", τ=f"{criterion.temperature.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for texts, voxels, _categories in tqdm(loader, desc="  val  ", leave=False):
        voxels = voxels.to(device)
        text_emb, voxel_emb = model(texts, voxels)
        loss = criterion(text_emb, voxel_emb)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train(cfg: dict):
    """Full training run."""
    set_seed(cfg["data"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    # --- data ---
    train_loader, val_loader, test_loader, block_mapping, num_blocks = \
        create_dataloaders(cfg)

    # --- model ---
    model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
    criterion = CLIPLoss(
        temperature_init=cfg["training"]["temperature_init"]
    ).to(device)

    # count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters — trainable: {trainable:,}, frozen: {frozen:,}")

    # --- optimizer ---
    train_cfg = cfg["training"]
    param_groups = [
        {"params": model.voxel_encoder.parameters(), "lr": train_cfg["lr_voxel"]},
        {"params": model.text_encoder.project.parameters(), "lr": train_cfg["lr_text_proj"]},
        {"params": criterion.parameters(), "lr": train_cfg["lr_voxel"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    total_steps = len(train_loader) * train_cfg["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=1e-6,
    )

    # --- training loop ---
    ckpt_dir = train_cfg["checkpoint_dir"]
    patience = train_cfg["early_stopping_patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Total steps: {total_steps}")
    print()

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"τ={criterion.temperature.item():.4f}  "
            f"lr={lr_current:.2e}  "
            f"time={elapsed:.1f}s"
        )

        scheduler.step()

        # checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "block_mapping": block_mapping,
                    "cfg": cfg,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # save final
    save_checkpoint(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "criterion_state": criterion.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "block_mapping": block_mapping,
            "cfg": cfg,
        },
        os.path.join(ckpt_dir, "last.pt"),
    )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, criterion, test_loader, block_mapping


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal MC retrieval model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
