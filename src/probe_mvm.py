"""Probe the pretrained MVM model: reconstruction accuracy at various mask ratios."""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from pretrain import MaskedVoxelModel, VoxelOnlyDataset, create_mask
from dataset import build_block_mapping, remap_voxel
from utils import load_config, set_seed, get_device
import pandas as pd


@torch.no_grad()
def probe_reconstruction(model, loader, device, mask_ratios=[0.1, 0.2, 0.3, 0.5, 0.7]):
    """Evaluate reconstruction accuracy at different mask ratios."""
    model.eval()

    for ratio in mask_ratios:
        total_correct = 0
        total_masked = 0
        total_correct_nonair = 0
        total_nonair = 0
        total_loss = 0
        n_batches = 0

        # per-block accuracy tracking
        block_correct = {}
        block_total = {}

        for voxels in tqdm(loader, desc=f"  mask={ratio:.0%}", leave=False):
            voxels = voxels.to(device)

            # manually create mask at this ratio
            mask = create_mask(voxels, mask_ratio=ratio)
            masked_voxels = voxels.clone()
            masked_voxels[mask] = model.mask_token_id

            # encode-decode
            x = model.block_embedding(masked_voxels)
            x = x.permute(0, 4, 1, 2, 3).contiguous()

            e1 = model.enc1(x)
            e2 = model.enc2(model.pool1(e1))
            bn = model.bottleneck(model.pool2(e2))

            d2 = model.up2(bn)
            d2 = model.dec2(torch.cat([d2, e2], dim=1))
            d1 = model.up1(d2)
            d1 = model.dec1(torch.cat([d1, e1], dim=1))

            logits = model.pred_head(d1)

            # loss
            loss = F.cross_entropy(logits, voxels, reduction="none")
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
            total_loss += loss.item()
            n_batches += 1

            # accuracy on masked positions
            preds = logits.argmax(dim=1)
            correct_mask = (preds == voxels) & mask
            total_correct += correct_mask.sum().item()
            total_masked += mask.sum().item()

            # accuracy on non-air masked positions
            nonair_mask = mask & (voxels != 0)
            total_correct_nonair += ((preds == voxels) & nonair_mask).sum().item()
            total_nonair += nonair_mask.sum().item()

            # per-block-type accuracy (on masked positions)
            for block_id in voxels[mask].unique().tolist():
                block_mask = mask & (voxels == block_id)
                if block_mask.sum() == 0:
                    continue
                c = ((preds == voxels) & block_mask).sum().item()
                t = block_mask.sum().item()
                block_correct[block_id] = block_correct.get(block_id, 0) + c
                block_total[block_id] = block_total.get(block_id, 0) + t

        acc = total_correct / max(total_masked, 1)
        acc_nonair = total_correct_nonair / max(total_nonair, 1)
        avg_loss = total_loss / max(n_batches, 1)

        print(f"\n  Mask ratio: {ratio:.0%}")
        print(f"    Loss:              {avg_loss:.4f}")
        print(f"    Overall accuracy:  {acc:.4f}  ({total_correct}/{total_masked})")
        print(f"    Non-air accuracy:  {acc_nonair:.4f}  ({total_correct_nonair}/{total_nonair})")

        # top-10 and bottom-10 block types by accuracy
        block_accs = {}
        for bid in block_total:
            if block_total[bid] >= 50:  # min samples
                block_accs[bid] = block_correct[bid] / block_total[bid]

        if block_accs and ratio == 0.2:  # only print detailed breakdown for 20%
            sorted_blocks = sorted(block_accs.items(), key=lambda x: x[1], reverse=True)
            print(f"\n    Top-10 block types (by recon accuracy):")
            for bid, bacc in sorted_blocks[:10]:
                print(f"      block {bid:3d}: {bacc:.4f}  (n={block_total[bid]})")
            print(f"    Bottom-10 block types:")
            for bid, bacc in sorted_blocks[-10:]:
                print(f"      block {bid:3d}: {bacc:.4f}  (n={block_total[bid]})")


def main():
    parser = argparse.ArgumentParser(description="Probe MVM reconstruction")
    parser.add_argument("--config", type=str, default="configs/cnn/cnn_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrained_voxel.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device()

    # load data (use test split for clean eval)
    df = pd.read_parquet(cfg["data"]["parquet_path"])
    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=cfg["data"]["max_block_types"]
    )

    # use last 10% as test
    n = len(df)
    test_df = df.iloc[int(n * 0.9):]
    print(f"Probing on {len(test_df)} samples")

    dataset = VoxelOnlyDataset(test_df, block_mapping)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2,
                        pin_memory=True)

    # load model
    model_cfg = cfg["model"]
    num_blocks = cfg["data"]["max_block_types"]
    model = MaskedVoxelModel(
        num_block_types=num_blocks,
        block_embed_dim=model_cfg["block_embed_dim"],
        channels=model_cfg["voxel_channels"],
        dropout=0.0,  # no dropout for eval
        mask_ratio=0.2,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, train_acc={ckpt['accuracy']:.4f})")

    print("\n" + "=" * 60)
    print("MVM RECONSTRUCTION PROBE")
    print("=" * 60)

    probe_reconstruction(model, loader, device,
                         mask_ratios=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])


if __name__ == "__main__":
    main()
