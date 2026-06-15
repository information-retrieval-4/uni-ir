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


def extract_block_names(df: pd.DataFrame, block_mapping: dict) -> list[str]:
    """Extract string names for each raw ID in the block_mapping."""
    raw_id_to_name = {0: "air"}
    
    needed_ids = set(block_mapping.keys())
    needed_ids.remove(0)
    
    if "voxel_name_data" not in df.columns:
        return ["unknown block"] * len(block_mapping)
        
    for _, row in df.iterrows():
        vd = np.asarray(row["voxel_data"]).flatten()
        vnd = np.asarray(row["voxel_name_data"]).flatten()
        
        for i in range(len(vd)):
            raw_id = vd[i]
            if raw_id in needed_ids and raw_id not in raw_id_to_name:
                raw_id_to_name[raw_id] = vnd[i]
                
            if len(raw_id_to_name) == len(block_mapping):
                break
        if len(raw_id_to_name) == len(block_mapping):
            break
            
    compact_id_to_name = {compact_id: "unknown block" for compact_id in block_mapping.values()}
    compact_id_to_name[1] = "unknown block" # <rare>
    for raw_id, compact_id in block_mapping.items():
        if raw_id in raw_id_to_name:
            name = str(raw_id_to_name[raw_id]).replace("_", " ")
            if not name.endswith("block") and name != "air":
                name += " block"
            compact_id_to_name[compact_id] = name
            
    return [compact_id_to_name[i] for i in range(len(compact_id_to_name))]


def remap_voxel(voxel_flat, mapping: dict, crop_bbox: bool = True,
                target_size: int = 32) -> torch.LongTensor:
    """Remap a flat voxel array, optionally crop to bbox and resize to target³."""
    arr = np.asarray(voxel_flat, dtype=np.int64)
    remapped = np.array([mapping.get(v, 1) for v in arr], dtype=np.int64)
    vol = remapped.reshape(32, 32, 32)

    if crop_bbox:
        non_air = vol != 0
        if non_air.any():
            coords = np.argwhere(non_air)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            cropped = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

            # nearest-neighbor resize to target³ (block IDs are categorical)
            from scipy.ndimage import zoom
            shape = cropped.shape
            factors = (target_size / shape[0],
                       target_size / shape[1],
                       target_size / shape[2])
            vol = zoom(cropped, factors, order=0)  # order=0 = nearest neighbor

    return torch.from_numpy(vol.copy()).long()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SchematicDataset(Dataset):
    """Dataset of (text, voxel, category) tuples for contrastive learning."""

    def __init__(self, df: pd.DataFrame, block_mapping: dict, crop_bbox: bool = True, 
                 augment: bool = False, aug_apply_prob: float = 0.5, aug_dropout_prob: float = 0.05):
        self.texts = [build_text(row) for _, row in df.iterrows()]
        self.voxels = df["voxel_data"].tolist()
        self.categories = df["subtitle"].fillna("Unknown").tolist()
        self.block_mapping = block_mapping
        self.crop_bbox = crop_bbox
        self.augment = augment
        self.aug_apply_prob = aug_apply_prob
        self.aug_dropout_prob = aug_dropout_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        voxel = remap_voxel(self.voxels[idx], self.block_mapping, crop_bbox=self.crop_bbox)
        
        if self.augment:
            import random
            # 1. Random 90-degree rotations in the horizontal plane (assuming axes 0 and 2 are X and Z)
            k = random.randint(0, 3)
            if k > 0:
                voxel = torch.rot90(voxel, k, [0, 2])
                
            # 2. Block dropout
            if random.random() < self.aug_apply_prob:
                non_air_mask = voxel != 0
                drop_mask = torch.rand_like(voxel, dtype=torch.float) < self.aug_dropout_prob
                voxel[non_air_mask & drop_mask] = 0

        category = self.categories[idx]
        return text, voxel, category


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    cfg: dict,
    parquet_path: Optional[str] = None,
):
    """Load data, preprocess, split, and return train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, block_mapping, num_block_types, block_names
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
    block_names = extract_block_names(df, block_mapping)
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
    
    crop_bbox = data_cfg.get("crop_bbox", True)
    augment = data_cfg.get("augment", True)
    aug_apply_prob = data_cfg.get("aug_apply_prob", 0.5)
    aug_dropout_prob = data_cfg.get("aug_dropout_prob", 0.05)

    ds_train = SchematicDataset(df.iloc[idx_train].reset_index(drop=True), block_mapping, crop_bbox=crop_bbox, 
                                augment=augment, aug_apply_prob=aug_apply_prob, aug_dropout_prob=aug_dropout_prob)
    ds_val   = SchematicDataset(df.iloc[idx_val].reset_index(drop=True), block_mapping, crop_bbox=crop_bbox, augment=False)
    ds_test  = SchematicDataset(df.iloc[idx_test].reset_index(drop=True), block_mapping, crop_bbox=crop_bbox, augment=False)

    # --- loaders -------------------------------------------------------
    train_cfg = cfg["training"]

    def collate_fn(batch):
        texts, voxels, categories = zip(*batch)
        voxels = torch.stack(voxels)
        return list(texts), voxels, list(categories)

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

    return train_loader, val_loader, test_loader, block_mapping, num_block_types, block_names
