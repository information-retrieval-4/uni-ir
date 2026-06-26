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

from dataset import build_block_mapping, remap_voxel, VoxelOnlyDataset
from utils import load_config, set_seed, get_device, save_checkpoint
from model import DepthwiseSeparableConv3d


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
    non_air = voxels != 0  # (B, 32, 32, 32)
    rand = torch.rand_like(voxels, dtype=torch.float32)
    mask = non_air & (rand < mask_ratio)
    return mask


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
        embed_dim: int = 256,
        dropout: float = 0.3,
        mask_ratio: float = 0.2,
        use_learned_stem: bool = False,
        use_depthwise_separable: bool = False,
        use_depthwise_separable_decoder: bool = False,
    ):
        super().__init__()
        self.num_block_types = num_block_types
        self.mask_token_id = num_block_types  # extra token for [MASK]
        self.mask_ratio = mask_ratio
        self.use_learned_stem = use_learned_stem

        # +1 for mask token
        self.block_embedding = nn.Embedding(num_block_types + 1, block_embed_dim)

        conv_enc_cls = (
            DepthwiseSeparableConv3d if use_depthwise_separable else nn.Conv3d
        )
        conv_dec_cls = (
            DepthwiseSeparableConv3d if use_depthwise_separable_decoder else nn.Conv3d
        )

        # --- Encoder (mirrors VoxelEncoder.conv_stack) ---
        if use_learned_stem:
            self.stem = nn.Sequential(
                nn.Conv3d(block_embed_dim, block_embed_dim, 4, stride=2, padding=1),
                nn.BatchNorm3d(block_embed_dim),
                nn.GELU(),
            )
        enc_in = block_embed_dim

        # Block 1
        self.enc1 = nn.Sequential(
            conv_enc_cls(enc_in, channels[0], 3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool1 = nn.MaxPool3d(2)

        # Block 2
        self.enc2 = nn.Sequential(
            conv_enc_cls(channels[0], channels[1], 3, padding=1),
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck (no pooling)
        self.bottleneck = nn.Sequential(
            conv_enc_cls(channels[1], channels[2], 3, padding=1),
            nn.BatchNorm3d(channels[2]),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )

        # --- Decoder ---
        # Up 2: concat with enc2
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec2 = nn.Sequential(
            conv_dec_cls(channels[1] * 2, channels[1], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[1]),
            nn.GELU(),
        )

        # Up 1: concat with enc1
        self.up1 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec1 = nn.Sequential(
            conv_dec_cls(channels[0] * 2, channels[0], 3, padding=1),  # *2 for skip
            nn.BatchNorm3d(channels[0]),
            nn.GELU(),
        )

        if use_learned_stem:
            self.up_stem = nn.ConvTranspose3d(channels[0], block_embed_dim, 2, stride=2)
            self.dec_stem = nn.Sequential(
                conv_dec_cls(block_embed_dim * 2, block_embed_dim, 3, padding=1),
                nn.BatchNorm3d(block_embed_dim),
                nn.GELU(),
            )
            self.pred_head = nn.Conv3d(block_embed_dim, num_block_types, 1)
        else:
            self.pred_head = nn.Conv3d(channels[0], num_block_types, 1)

        # --- SimCLR Head ---
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.simclr_proj = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.Linear(channels[-1], embed_dim),
        )

    def forward(
        self,
        voxels: torch.LongTensor,
        return_simclr: bool = False,
        return_mvm: bool = True,
    ):
        """
        Args:
            voxels: (B, 32, 32, 32) original block IDs
        Returns:
            logits: (B, num_blocks, 32, 32, 32) predictions (or None if return_mvm=False)
            mask:   (B, 32, 32, 32) bool mask of what was masked (or None if return_mvm=False)
            (optional) simclr_emb: (B, embed_dim) if return_simclr is True
        """
        # Create mask and apply
        mask = create_mask(voxels, self.mask_ratio)
        masked_voxels = voxels.clone()
        masked_voxels[mask] = self.mask_token_id

        # Embed
        x = self.block_embedding(masked_voxels)  # (B, 32, 32, 32, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, 32, 32, 32)

        if self.use_learned_stem:
            e_in = self.stem(x)
        else:
            e_in = x

        # Encoder
        e1 = self.enc1(e_in)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))

        simclr_emb = None
        if return_simclr:
            simclr_emb = self.global_pool(bn).flatten(1)
            simclr_emb = self.simclr_proj(simclr_emb)
            simclr_emb = nn.functional.normalize(simclr_emb, dim=-1)

        if not return_mvm:
            return None, None, simclr_emb

        # Decoder with skip connections
        d2 = self.up2(bn)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        if self.use_learned_stem:
            d_stem = self.up_stem(d1)
            d_stem = self.dec_stem(torch.cat([d_stem, x], dim=1))
            logits = self.pred_head(d_stem)
        else:
            logits = self.pred_head(d1)

        if return_simclr:
            return logits, mask, simclr_emb

        return logits, mask

    def get_encoder_state_dict(self):
        """Extract encoder weights in VoxelEncoder-compatible format."""
        state = {}

        # Block embedding (drop mask token)
        state["block_embedding.weight"] = self.block_embedding.weight[
            : self.num_block_types
        ].clone()

        offset = 0
        if self.use_learned_stem:
            stem_conv = self.stem[0]
            stem_bn = self.stem[1]
            state["conv_stack.0.weight"] = stem_conv.weight.clone()
            state["conv_stack.0.bias"] = stem_conv.bias.clone()
            state["conv_stack.1.weight"] = stem_bn.weight.clone()
            state["conv_stack.1.bias"] = stem_bn.bias.clone()
            state["conv_stack.1.running_mean"] = stem_bn.running_mean.clone()
            state["conv_stack.1.running_var"] = stem_bn.running_var.clone()
            state["conv_stack.1.num_batches_tracked"] = (
                stem_bn.num_batches_tracked.clone()
            )
            offset = 3

        mapping = {
            "enc1": offset + 0,
            "enc2": offset + 5,
            "bottleneck": offset + 10,
        }

        for block_name, stack_offset in mapping.items():
            block = getattr(self, block_name)
            conv = block[0]

            if isinstance(conv, DepthwiseSeparableConv3d):
                state[f"conv_stack.{stack_offset}.depthwise.weight"] = (
                    conv.depthwise.weight.clone()
                )
                state[f"conv_stack.{stack_offset}.depthwise.bias"] = (
                    conv.depthwise.bias.clone()
                )
                state[f"conv_stack.{stack_offset}.pointwise.weight"] = (
                    conv.pointwise.weight.clone()
                )
                state[f"conv_stack.{stack_offset}.pointwise.bias"] = (
                    conv.pointwise.bias.clone()
                )
            else:
                state[f"conv_stack.{stack_offset}.weight"] = conv.weight.clone()
                state[f"conv_stack.{stack_offset}.bias"] = conv.bias.clone()

            bn = block[1]
            state[f"conv_stack.{stack_offset + 1}.weight"] = bn.weight.clone()
            state[f"conv_stack.{stack_offset + 1}.bias"] = bn.bias.clone()
            state[f"conv_stack.{stack_offset + 1}.running_mean"] = (
                bn.running_mean.clone()
            )
            state[f"conv_stack.{stack_offset + 1}.running_var"] = bn.running_var.clone()
            state[f"conv_stack.{stack_offset + 1}.num_batches_tracked"] = (
                bn.num_batches_tracked.clone()
            )

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

    pretrain_mode = pt_cfg.get("mode", "mvm")  # "mvm", "simclr", "hybrid"
    mvm_weight = pt_cfg.get("mvm_weight", 1.0)
    simclr_weight = pt_cfg.get("simclr_weight", 1.0)
    num_views = 2 if pretrain_mode in ["simclr", "hybrid"] else 1

    # --- data (use ALL samples, no splits needed) ---
    df = pd.read_parquet(cfg["data"]["parquet_path"])
    print(f"Loaded {len(df)} samples for pretraining")

    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )
    num_blocks = cfg["data"]["max_block_types"]

    crop_bbox = cfg["data"].get("crop_bbox", True)
    augment = pt_cfg.get("augment", cfg["data"].get("augment", True))
    aug_apply_prob = cfg["data"].get("aug_apply_prob", 0.5)
    aug_dropout_prob = cfg["data"].get("aug_dropout_prob", 0.05)

    dataset = VoxelOnlyDataset(
        df,
        block_mapping,
        crop_bbox=crop_bbox,
        augment=augment,
        aug_apply_prob=aug_apply_prob,
        aug_dropout_prob=aug_dropout_prob,
        num_views=num_views,
    )
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
        embed_dim=model_cfg["embed_dim"],
        dropout=model_cfg.get("dropout", 0.3),
        mask_ratio=mask_ratio,
        use_learned_stem=model_cfg.get("use_learned_stem", False),
        use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
        use_depthwise_separable_decoder=model_cfg.get(
            "use_depthwise_separable_decoder", False
        ),
    ).to(device)

    if model_cfg.get("semantic_init", False):
        from dataset import extract_block_names
        from model import TextEncoder, apply_semantic_init

        block_names = extract_block_names(df, block_mapping)
        # mask token gets an arbitrary string "mask token"
        block_names.append("mask token")

        if model_cfg.get("use_trimodal", False):
            import open_clip
            arch = model_cfg.get("tinyclip_arch", "TinyCLIP-auto-ViT-45M-32-Text-18M")
            pretrained = model_cfg.get("tinyclip_pretrained", "LAIONYFCC400M")
            print(f"Loading TinyCLIP for semantic init: {arch} ({pretrained})")
            clip_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)
            clip_model = clip_model.to(device)

            class TinyCLIPWrapper:
                def encode_text(self, texts):
                    tokens = tokenizer(texts).to(device)
                    with torch.no_grad():
                        emb = clip_model.encode_text(tokens)
                    return emb

            temp_text_encoder = TinyCLIPWrapper()
        else:
            temp_text_encoder = TextEncoder(
                model_name=model_cfg["text_model"],
                text_hidden_dim=model_cfg["text_hidden_dim"],
                embed_dim=model_cfg["embed_dim"],
                freeze=True,
            ).to(device)

        apply_semantic_init(
            voxel_embedding_layer=model.block_embedding,
            text_encoder=temp_text_encoder,
            block_names=block_names,
            block_embed_dim=model_cfg["block_embed_dim"],
            device=device,
        )
        
        if model_cfg.get("use_trimodal", False):
            del clip_model
            del tokenizer
            del temp_text_encoder
        else:
            del temp_text_encoder
        torch.cuda.empty_cache()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"MVM model parameters: {param_count:,}")

    # --- optimizer ---
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # --- training ---
    print(f"\nStarting {pretrain_mode.upper()} pretraining for {epochs} epochs...")
    print(f"  Mask ratio: {mask_ratio}")
    print(f"  Batch size: {batch_size}")
    if pretrain_mode in ["simclr", "hybrid"]:
        from losses import SimCLRLoss

        simclr_criterion = SimCLRLoss(temperature_init=0.1).to(device)
    print()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mvm_loss = 0.0
        total_simclr_loss = 0.0
        total_correct = 0
        total_masked = 0
        num_batches = 0

        pbar = tqdm(loader, desc=f"  epoch {epoch:3d}", leave=False)
        for batch in pbar:
            if num_views == 2:
                voxels1, voxels2 = batch
                voxels1, voxels2 = voxels1.to(device), voxels2.to(device)
            else:
                voxels = batch.to(device)

            optimizer.zero_grad()
            loss = 0.0

            if pretrain_mode == "mvm":
                logits, mask = model(voxels)
                mvm_loss = F.cross_entropy(logits, voxels, reduction="none")
                mvm_loss = (mvm_loss * mask.float()).sum() / mask.float().sum().clamp(
                    min=1
                )
                loss = mvm_loss

                preds = logits.argmax(dim=1)
                correct = ((preds == voxels) & mask).sum().item()
                n_masked = mask.sum().item()

                total_mvm_loss += mvm_loss.item()
                total_correct += correct
                total_masked += n_masked

            elif pretrain_mode == "simclr":
                old_mask_ratio = model.mask_ratio
                model.mask_ratio = 0.0
                _, _, z1 = model(voxels1, return_simclr=True, return_mvm=False)
                _, _, z2 = model(voxels2, return_simclr=True, return_mvm=False)
                model.mask_ratio = old_mask_ratio

                simclr_loss = simclr_criterion(z1, z2)
                loss = simclr_loss

                total_simclr_loss += simclr_loss.item()

            elif pretrain_mode == "hybrid":
                # only run the decoder (MVM) on view 1
                logits1, mask1, z1 = model(voxels1, return_simclr=True, return_mvm=True)
                _, _, z2 = model(voxels2, return_simclr=True, return_mvm=False)

                mvm_loss = F.cross_entropy(logits1, voxels1, reduction="none")
                mvm_loss = (mvm_loss * mask1.float()).sum() / mask1.float().sum().clamp(
                    min=1
                )

                simclr_loss = simclr_criterion(z1, z2)

                loss = (mvm_weight * mvm_loss) + (simclr_weight * simclr_loss)

                preds = logits1.argmax(dim=1)
                correct = ((preds == voxels1) & mask1).sum().item()
                n_masked = mask1.sum().item()

                total_mvm_loss += mvm_loss.item()
                total_simclr_loss += simclr_loss.item()
                total_correct += correct
                total_masked += n_masked

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if pretrain_mode == "mvm":
                acc = correct / max(n_masked, 1)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")
            elif pretrain_mode == "simclr":
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            else:
                acc = correct / max(n_masked, 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mvm=f"{mvm_loss.item():.3f}",
                    sim=f"{simclr_loss.item():.3f}",
                )

        scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = (
            total_correct / max(total_masked, 1) if pretrain_mode != "simclr" else 0.0
        )
        lr_current = optimizer.param_groups[0]["lr"]

        if pretrain_mode == "simclr":
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"lr={lr_current:.2e}"
            )
        else:
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
    parser = argparse.ArgumentParser(
        description="Pretrain voxel encoder via masked voxel modeling"
    )
    parser.add_argument("--config", type=str, default="configs/cnn/cnn_default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pretrain(cfg)
