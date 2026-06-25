"""Training loop for the dual encoder model."""

import os
import time
import argparse
import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder, TrimodalEncoder
from losses import CLIPLoss, NTXentLoss
from utils import load_config, set_seed, get_device, save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, device, augment=True, use_amp=False
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  train", leave=False)
    for batch in pbar:
        if len(batch) == 4:
            texts, voxels, images, _categories = batch
            images = images.to(device)
        else:
            texts, voxels, _categories = batch
            images = None

        voxels = voxels.to(device)

        if augment:
            voxels = augment_voxels(voxels)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            if images is not None:
                text_emb, voxel_emb, image_emb = model(texts, voxels, images)
                loss = (criterion(text_emb, voxel_emb) + criterion(image_emb, voxel_emb) + criterion(text_emb, image_emb)) / 3.0
            else:
                text_emb, voxel_emb = model(texts, voxels)
                loss = criterion(text_emb, voxel_emb)

        loss.backward()
        all_params = list(model.parameters()) + list(criterion.parameters())
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # log temp if using CLIPLoss, otherwise just loss
        if hasattr(criterion, 'temperature'):
            temp = criterion.temperature.item() if isinstance(criterion.temperature, torch.Tensor) else criterion.temperature
            pbar.set_postfix(loss=f"{loss.item():.4f}", τ=f"{temp:.4f}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        if len(batch) == 4:
            texts, voxels, images, _categories = batch
            images = images.to(device)
        else:
            texts, voxels, _categories = batch
            images = None
            
        voxels = voxels.to(device)
        
        with autocast('cuda', enabled=use_amp):
            if images is not None:
                text_emb, voxel_emb, image_emb = model(texts, voxels, images)
                loss = (criterion(text_emb, voxel_emb) + criterion(image_emb, voxel_emb) + criterion(text_emb, image_emb)) / 3.0
            else:
                text_emb, voxel_emb = model(texts, voxels)
                loss = criterion(text_emb, voxel_emb)
                
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def build_scheduler(optimizer, cfg: dict):
    tr_cfg       = cfg["training"]
    total_epochs = tr_cfg["epochs"]
    warmup_ratio = tr_cfg.get("warmup_ratio", 0.05)
    
    if warmup_ratio > 0:
        warmup_steps = max(1, math.ceil(total_epochs * warmup_ratio))
        warmup = LinearLR(
            optimizer,
            start_factor = 1e-3,
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
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=1e-6,
        )
        print("  Scheduler: cosine annealing (no warmup)")
        return scheduler


def train(cfg: dict, pretrained_path: str = None, warmstart_path: str = None):
    """Full training run."""
    set_seed(cfg["data"]["seed"])
    device = get_device()
    use_amp = cfg["training"].get("use_amp", False) and device.type == "cuda"
    print(f"Device: {device}")
    print(f"AMP   : {'enabled' if use_amp else 'disabled'}")

    # --- model initialization ---
    use_trimodal = cfg.get("model", {}).get("use_trimodal", False)
    num_blocks = cfg["data"]["max_block_types"]
    
    if use_trimodal:
        model = TrimodalEncoder(cfg, num_block_types=num_blocks).to(device)
        image_preprocess = getattr(model, "preprocess", None)
    else:
        model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
        image_preprocess = None

    # --- data ---
    train_loader, val_loader, test_loader, block_mapping, num_blocks, block_names = (
        create_dataloaders(cfg, image_preprocess=image_preprocess)
    )

    # --- load pretrained voxel encoder weights (for CNN typically) ---
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained voxel encoder from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        encoder_state = ckpt["encoder_state"]
        missing, unexpected = model.voxel_encoder.load_state_dict(
            encoder_state, strict=False
        )
        print(f"  Loaded pretrained weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (will train from scratch): {missing}")
            
    # --- warmstart (for PointBERT Plan 1 from Plan 2 usually) ---
    if warmstart_path and os.path.exists(warmstart_path):
        print(f"\n[WarmStart] Loading checkpoint: {warmstart_path}")
        ckpt = torch.load(warmstart_path, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[WarmStart] Missing  keys: {len(missing)}")
        print(f"[WarmStart] Unexpected keys: {len(unexpected)}\n")

    # --- Semantic init logic ---
    if cfg["model"].get("semantic_init", False) and not pretrained_path and not warmstart_path:
        from model import apply_semantic_init
        if getattr(model.voxel_encoder, "init_semantic_embeddings", None) is not None:
            # pointbert branch handles it internally with text encoder
            if "__index_to_name__" in block_mapping:
                print("\n[Strategy 2] Initializing PointBERT block_embedding with semantic names...")
                freeze_emb = cfg["model"].get("semantic_init_freeze", False)
                model.voxel_encoder.init_semantic_embeddings(
                    index_to_name  = block_mapping["__index_to_name__"],
                    sentence_model = model.text_encoder.encoder,
                    freeze         = freeze_emb,
                )
        else:
            # cnn branch
            apply_semantic_init(
                voxel_embedding_layer=model.voxel_encoder.block_embedding,
                text_encoder=model.text_encoder,
                block_names=block_names,
                block_embed_dim=cfg["model"]["block_embed_dim"],
                device=device,
            )

    if use_trimodal:
        criterion = NTXentLoss(temperature=cfg["training"].get("temperature_init", 0.07)).to(device)
    else:
        criterion = CLIPLoss(temperature_init=cfg["training"].get("temperature_init", 0.07)).to(device)

    # count params
    trainable, frozen = count_params(model)
    print(f"Parameters — trainable: {trainable:,}, frozen: {frozen:,}")

    # --- optimizer ---
    train_cfg = cfg["training"]
    param_groups = model.get_param_groups(cfg)
    param_groups.append({
        "params": list(criterion.parameters()),
        "lr": train_cfg.get("lr_adapter", train_cfg.get("lr_voxel", 1e-4)),
        "name": "temperature",
    })
    
    print_param_groups(param_groups)
    optimizer = AdamW(param_groups, weight_decay=train_cfg["weight_decay"])

    scheduler = build_scheduler(optimizer, cfg)

    # --- training loop ---
    ckpt_dir = train_cfg["checkpoint_dir"]
    patience = train_cfg["early_stopping_patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    total_steps = len(train_loader) * train_cfg["epochs"]

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Total steps: {total_steps}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_amp=use_amp
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        
        if hasattr(criterion, 'temperature'):
            temp = criterion.temperature.item() if isinstance(criterion.temperature, torch.Tensor) else criterion.temperature
            print(
                f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"τ={temp:.4f}  "
                f"lr={lr_current:.2e}  "
                f"time={elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
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
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
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
    parser.add_argument("--config", type=str, default="configs/cnn_default.yaml")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained voxel encoder checkpoint (from pretrain.py)",
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Optional: path to warm-start checkpoint (e.g. for PointBERT Plan 1)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Print active plan info if pointbert
    if cfg["model"].get("encoder_type") == "pointbert":
        is_frozen = cfg.get("pointbert", {}).get("freeze_backbone", True)
        plan_label = "❄️  Plan 2 — Feature Extraction (Frozen Backbone)" if is_frozen \
                else "🔥  Plan 1 — Full Fine-Tuning (Trainable Backbone)"
        print(f"\n{'='*60}")
        print(f"  {plan_label}")
        print(f"{'='*60}")
        
    train(cfg, pretrained_path=args.pretrained, warmstart_path=args.warmstart)
