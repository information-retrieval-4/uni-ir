"""Training script for Point-BERT dual-encoder (Plan 1 & Plan 2).

Usage:
    # Plan 2 (frozen backbone — mulai dari sini):
    python train_pointbert.py --config configs/pointbert.yaml

    # Plan 1 (full fine-tune):
    python train_pointbert.py --config configs/pointbert_finetune.yaml

    # Plan 1 dengan warm-start dari Plan 2 checkpoint:
    python train_pointbert.py --config configs/pointbert_finetune.yaml \\
        --warmstart checkpoints/pointbert_plan2/best.pt

Key differences from train.py:
  - Pakai DualEncoderPointBERT bukan DualEncoder
  - Param groups dengan discriminative LR (adapter >> backbone)
  - AMP (torch.cuda.amp) untuk hemat VRAM
  - Warmup scheduler (Linear warmup → CosineAnnealing)
  - Opsional: warm-start dari checkpoint Plan 2
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Pastikan src/ ada di PYTHONPATH jika dijalankan dari folder lain
sys.path.insert(0, os.path.dirname(__file__))

from dataset import create_dataloaders
from losses import CLIPLoss
from model_pointbert import DualEncoderPointBERT
from utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: parameter counting & group summary
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def print_param_groups(groups: list):
    print("\n  Optimizer Parameter Groups:")
    total_trainable = 0
    for g in groups:
        n = sum(p.numel() for p in g["params"] if isinstance(p, nn.Parameter) or isinstance(p, torch.Tensor))
        print(f"    [{g.get('name', '?'):20s}]  lr={g['lr']:.1e}   params={n:,}")
        total_trainable += n
    print(f"    {'TOTAL TRAINABLE':20s}           {total_trainable:,}\n")


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    use_amp: bool = True,
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for texts, voxels, _cats in pbar:
        voxels = voxels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            text_emb, voxel_emb = model(texts, voxels)
            loss = criterion(text_emb, voxel_emb)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss  += loss.item()
        num_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            τ=f"{criterion.temperature.item():.4f}",
        )

    return total_loss / max(num_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp: bool = True):
    model.eval()
    total_loss  = 0.0
    num_batches = 0

    for texts, voxels, _cats in tqdm(loader, desc="  val  ", leave=False):
        voxels = voxels.to(device)
        with autocast(enabled=use_amp):
            text_emb, voxel_emb = model(texts, voxels)
            loss = criterion(text_emb, voxel_emb)
        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler: Linear Warmup → Cosine Annealing
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict):
    tr_cfg       = cfg["training"]
    total_epochs = tr_cfg["epochs"]
    warmup_ratio = tr_cfg.get("warmup_ratio", 0.05)
    warmup_steps = max(1, math.ceil(total_epochs * warmup_ratio))

    warmup = LinearLR(
        optimizer,
        start_factor = 1e-3,   # mulai dari LR yang sangat kecil
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = total_epochs - warmup_steps,
        eta_min = 1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_steps],
    )
    print(f"  Scheduler: {warmup_steps} epoch warmup → cosine annealing")
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict, warmstart_path: str = None):
    """Full training loop for Point-BERT dual encoder."""

    set_seed(cfg["data"]["seed"])
    device  = get_device()
    use_amp = cfg["training"].get("use_amp", True) and device.type == "cuda"
    print(f"\nDevice : {device}")
    print(f"AMP    : {'enabled' if use_amp else 'disabled (CPU)'}\n")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, block_mapping, num_blocks = \
        create_dataloaders(cfg)

    # ── Model ──────────────────────────────────────────────────────────────
    model     = DualEncoderPointBERT(cfg, num_block_types=num_blocks).to(device)
    criterion = CLIPLoss(
        temperature_init=cfg["training"]["temperature_init"]
    ).to(device)

    # ── Optional warm-start from Plan 2 checkpoint ─────────────────────────
    if warmstart_path and os.path.exists(warmstart_path):
        print(f"\n[WarmStart] Loading Plan-2 checkpoint: {warmstart_path}")
        ckpt = load_checkpoint(warmstart_path, device)
        # Load only voxel encoder weights (adapter + backbone)
        missing, unexpected = model.load_state_dict(
            ckpt["model_state"], strict=False
        )
        print(f"[WarmStart] Missing  keys: {len(missing)}")
        print(f"[WarmStart] Unexpected keys: {len(unexpected)}\n")

    # ── Param counts ───────────────────────────────────────────────────────
    trainable, frozen = count_params(model)
    print(f"Parameters — trainable: {trainable:,}  |  frozen: {frozen:,}\n")

    # ── Optimizer (discriminative LR) ──────────────────────────────────────
    tr_cfg      = cfg["training"]
    param_groups = model.param_groups(cfg)

    # Add CLIPLoss temperature to optimizer (always trainable)
    param_groups.append({
        "params": criterion.parameters(),
        "lr": tr_cfg.get("lr_adapter", 3e-4),
        "name": "temperature",
    })

    print_param_groups(param_groups)
    optimizer = AdamW(param_groups, weight_decay=tr_cfg["weight_decay"])

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler = build_scheduler(optimizer, cfg)

    # ── AMP Scaler ─────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ──────────────────────────────────────────────────────
    ckpt_dir        = tr_cfg["checkpoint_dir"]
    patience        = tr_cfg["early_stopping_patience"]
    best_val_loss   = float("inf")
    epochs_no_imprv = 0

    print(f"Starting training for {tr_cfg['epochs']} epochs")
    print(f"  Batch size : {tr_cfg['batch_size']}")
    print(f"  Patience   : {patience}\n")

    for epoch in range(1, tr_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{tr_cfg['epochs']}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"τ={criterion.temperature.item():.4f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            epochs_no_imprv = 0
            save_checkpoint(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "criterion_state":criterion.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "val_loss":       val_loss,
                    "block_mapping":  block_mapping,
                    "cfg":            cfg,
                },
                os.path.join(ckpt_dir, "best.pt"),
            )
        else:
            epochs_no_imprv += 1
            if epochs_no_imprv >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Save last checkpoint
    save_checkpoint(
        {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "criterion_state":criterion.state_dict(),
            "val_loss":       val_loss,
            "block_mapping":  block_mapping,
            "cfg":            cfg,
        },
        os.path.join(ckpt_dir, "last.pt"),
    )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, criterion, test_loader, block_mapping


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Point-BERT dual-encoder (Plan 1 or Plan 2)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/pointbert.yaml",
        help="Path to YAML config (pointbert.yaml=Plan2, pointbert_finetune.yaml=Plan1)",
    )
    parser.add_argument(
        "--warmstart", type=str, default=None,
        help="Optional: path to Plan-2 checkpoint to warm-start Plan-1 training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Print active plan
    is_frozen = cfg.get("pointbert", {}).get("freeze_backbone", True)
    plan_label = "❄️  Plan 2 — Feature Extraction (Frozen Backbone)" if is_frozen \
            else "🔥  Plan 1 — Full Fine-Tuning (Trainable Backbone)"
    print(f"\n{'='*60}")
    print(f"  {plan_label}")
    print(f"{'='*60}")

    train(cfg, warmstart_path=args.warmstart)
