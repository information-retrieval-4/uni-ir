"""Dataset and preprocessing for Minecraft schematic retrieval."""

import json
import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def build_text(row: pd.Series) -> str:
    """Concatenate text fields into a single retrieval string."""
    parts = []
    for field in ("title", "subtitle", "description"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(clean_text(val))

    # tags may be a JSON list string
    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tag_list = json.loads(tags)
            if isinstance(tag_list, list):
                parts.append(", ".join(str(t) for t in tag_list))
        except (json.JSONDecodeError, TypeError):
            parts.append(clean_text(tags))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Voxel preprocessing
# ---------------------------------------------------------------------------

def build_block_mapping(voxel_series: pd.Series, max_types: int = 256):
    """Build a mapping from raw block IDs → compact indices.

    Keeps the top `max_types - 2` most frequent non-air blocks.
    Index 0 = air (raw ID 0), index 1 = <rare>, rest = frequent blocks.
    Returns: dict {raw_id: compact_id}
    """
    counter: Counter = Counter()
    for vd in voxel_series:
        arr = np.asarray(vd)
        counter.update(arr.tolist())

    # air is always 0
    if 0 in counter:
        del counter[0]

    # keep top-(max_types - 2) non-air blocks  (reserve 0=air, 1=rare)
    top_blocks = [block_id for block_id, _ in counter.most_common(max_types - 2)]
    mapping = {0: 0}  # air → 0
    for i, bid in enumerate(top_blocks, start=2):
        mapping[bid] = i
    # everything else maps to 1 (<rare>)
    return mapping


def remap_voxel(voxel_flat, mapping: dict) -> torch.LongTensor:
    """Remap a flat voxel array using the block mapping and reshape to 32³."""
    arr = np.asarray(voxel_flat, dtype=np.int64)
    remapped = np.array([mapping.get(v, 1) for v in arr], dtype=np.int64)
    return torch.from_numpy(remapped).reshape(32, 32, 32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SchematicDataset(Dataset):
    """Dataset of (text, voxel) pairs for contrastive learning."""

    def __init__(self, df: pd.DataFrame, block_mapping: dict):
        self.texts = [build_text(row) for _, row in df.iterrows()]
        self.voxels = df["voxel_data"].tolist()
        self.block_mapping = block_mapping

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        voxel = remap_voxel(self.voxels[idx], self.block_mapping)
        return text, voxel


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    cfg: dict,
    parquet_path: Optional[str] = None,
):
    """Load data, preprocess, split, and return train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, block_mapping, num_block_types
    """
    data_cfg = cfg["data"]
    path = parquet_path or data_cfg["parquet_path"]

    # --- load ----------------------------------------------------------
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} samples from {path}")

    # --- block mapping -------------------------------------------------
    block_mapping = build_block_mapping(
        df["voxel_data"], max_types=data_cfg["max_block_types"]
    )
    num_block_types = data_cfg["max_block_types"]
    print(f"Block vocabulary: {num_block_types} types "
          f"(mapped from {len(set().union(*[set(np.asarray(v).tolist()) for v in df['voxel_data'].head(100)]))}+ unique raw IDs)")

    # --- splits --------------------------------------------------------
    seed = data_cfg["seed"]
    val_frac = data_cfg["val_split"]
    test_frac = data_cfg["test_split"]

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=seed
    )
    relative_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=relative_test, random_state=seed
    )

    print(f"Splits — train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    ds_train = SchematicDataset(df.iloc[idx_train].reset_index(drop=True), block_mapping)
    ds_val   = SchematicDataset(df.iloc[idx_val].reset_index(drop=True), block_mapping)
    ds_test  = SchematicDataset(df.iloc[idx_test].reset_index(drop=True), block_mapping)

    # --- loaders -------------------------------------------------------
    train_cfg = cfg["training"]

    def collate_fn(batch):
        texts, voxels = zip(*batch)
        voxels = torch.stack(voxels)
        return list(texts), voxels

    train_loader = DataLoader(
        ds_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,      # important for contrastive — don't want tiny last batch
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, block_mapping, num_block_types
