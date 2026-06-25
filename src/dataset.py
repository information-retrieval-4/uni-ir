"""Dataset and preprocessing for Minecraft schematic retrieval.

Supports two parquet formats:
  - Old (data.parquet):                              uses voxel_data (numeric IDs)
  - New (data_with_voxel_names_multiview_image.parquet): uses voxel_name_data (strings)

Strategy flags (all under cfg['data']):
  use_name_vocab      : bool — Strategy 1: name-based block vocabulary
  use_material_context: bool — Strategy 3: append top-K block names to text
  top_k_materials     : int  — how many dominant block names to append (default 5)
"""

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
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text(row: pd.Series) -> str:
    """Concatenate text fields into a single retrieval string."""
    parts = []
    for field in ("title", "subtitle", "description"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(clean_text(val))

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
# Voxel preprocessing — Original (numeric ID based)
# ---------------------------------------------------------------------------

def build_block_mapping(voxel_series: pd.Series, max_types: int = 256):
    """Build a mapping from raw block IDs to compact indices.

    Keeps top (max_types - 2) most frequent non-air blocks.
    Index 0 = air, index 1 = <rare>, index 2..N = frequent blocks.
    Returns: dict {raw_id: compact_id}
    """
    counter: Counter = Counter()
    for vd in voxel_series:
        arr = np.asarray(vd)
        counter.update(arr.tolist())

    if 0 in counter:
        del counter[0]

    top_blocks = [block_id for block_id, _ in counter.most_common(max_types - 2)]
    mapping = {0: 0}
    for i, bid in enumerate(top_blocks, start=2):
        mapping[bid] = i
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
    """Remap flat numeric voxel array, optionally crop bbox and resize."""
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

            from scipy.ndimage import zoom
            shape = cropped.shape
            factors = (target_size / shape[0],
                       target_size / shape[1],
                       target_size / shape[2])
            vol = zoom(cropped, factors, order=0)

    return torch.from_numpy(vol.copy()).long()


# ---------------------------------------------------------------------------
# Strategy 1: Name-Based Block Vocabulary
# ---------------------------------------------------------------------------

def build_block_name_mapping(name_series: pd.Series, max_types: int = 256) -> dict:
    """Build compact index mapping from Minecraft block NAME strings.

    Strategy 1: uses human-readable block names ('oak_log', 'stone_brick')
    instead of opaque numeric IDs. Naturally groups semantically similar blocks
    (all 'log' variants cluster near each other in embedding space).

    Index assignment:
      0    = 'air'
      1    = '<rare>' (blocks outside the top-(max_types-2))
      2..N = top-(max_types-2) most frequent non-air block names

    The special key '__index_to_name__' stores list[str] mapping compact_idx to
    block name, used by Strategy 2 for semantic embedding initialization.

    Returns: dict {block_name_str: compact_index, '__index_to_name__': list[str]}
    """
    counter: Counter = Counter()
    for name_arr in name_series:
        arr = np.asarray(name_arr, dtype=str)
        non_air = arr[(arr != "air") & (arr != "") & (arr != "nan")]
        counter.update(non_air.tolist())

    top_names = [name for name, _ in counter.most_common(max_types - 2)]

    mapping: dict = {"air": 0}
    index_to_name = ["air", "<rare>"]

    for i, name in enumerate(top_names, start=2):
        mapping[name] = i
        index_to_name.append(name)

    while len(index_to_name) < max_types:
        index_to_name.append("<pad>")

    mapping["__index_to_name__"] = index_to_name
    return mapping


def remap_voxel_from_names(
    name_flat,
    name_mapping: dict,
    crop_bbox: bool = True,
    target_size: int = 32,
) -> torch.LongTensor:
    """Remap flat string block-name array to compact integer indices.

    Strategy 1 counterpart to remap_voxel().
    Unknown block names map to index 1 (<rare>).
    """
    arr = np.asarray(name_flat, dtype=str)
    remapped = np.array(
        [name_mapping.get(n, 1) for n in arr], dtype=np.int64
    )
    vol = remapped.reshape(32, 32, 32)

    if crop_bbox:
        non_air = vol != 0
        if non_air.any():
            coords = np.argwhere(non_air)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            cropped = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

            from scipy.ndimage import zoom
            shape = cropped.shape
            factors = (target_size / shape[0],
                       target_size / shape[1],
                       target_size / shape[2])
            vol = zoom(cropped, factors, order=0)

    return torch.from_numpy(vol.copy()).long()


# ---------------------------------------------------------------------------
# Strategy 3: Material Context Text Augmentation
# ---------------------------------------------------------------------------

def build_text_with_materials(
    row: pd.Series,
    top_k_materials: int = 5,
    name_field: str = "voxel_name_data",
) -> str:
    """Concatenate metadata text + dominant block material names.

    Appends suffix: "[Materials: spruce_log, oak_planks, stone_brick, ...]"

    Enables TextEncoder to match material-level queries like "wooden house"
    or "stone castle" without any voxel access at query time.

    Args:
        row            : DataFrame row (must contain voxel_name_data)
        top_k_materials: number of dominant non-air block names to include
        name_field     : column name for the string voxel name array
    """
    base_text = build_text(row)

    name_arr = row.get(name_field)
    if name_arr is None:
        return base_text

    try:
        arr = np.asarray(name_arr, dtype=str)
        non_air = arr[(arr != "air") & (arr != "") & (arr != "nan")]
        if len(non_air) == 0:
            return base_text

        counter = Counter(non_air.tolist())
        top_materials = [name for name, _ in counter.most_common(top_k_materials)]
        if top_materials:
            base_text = base_text + f" [Materials: {', '.join(top_materials)}]"
    except Exception:
        pass

    return base_text


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_voxel(
    voxel: torch.LongTensor,
    aug_apply_prob: float = 0.5,
    aug_dropout_prob: float = 0.05,
) -> torch.LongTensor:
    """Random 90° horizontal rotation + block dropout for a single voxel grid.

    Args:
        voxel:            (32, 32, 32) block-ID tensor
        aug_apply_prob:   probability to apply block dropout
        aug_dropout_prob: fraction of non-air blocks to zero out
    """
    import random
    k = random.randint(0, 3)
    if k > 0:
        voxel = torch.rot90(voxel, k, [0, 2])
    if random.random() < aug_apply_prob:
        non_air_mask = voxel != 0
        drop_mask = torch.rand_like(voxel, dtype=torch.float) < aug_dropout_prob
        voxel[non_air_mask & drop_mask] = 0
    return voxel


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class VoxelOnlyDataset(Dataset):
    """Dataset that returns only voxel grids for self-supervised pretraining."""

    def __init__(
        self,
        df: pd.DataFrame,
        block_mapping: dict,
        crop_bbox: bool = True,
        augment: bool = False,
        aug_apply_prob: float = 0.5,
        aug_dropout_prob: float = 0.05,
        num_views: int = 1,
    ):
        self.voxels = df["voxel_data"].tolist()
        self.block_mapping = block_mapping
        self.crop_bbox = crop_bbox
        self.augment = augment
        self.aug_apply_prob = aug_apply_prob
        self.aug_dropout_prob = aug_dropout_prob
        self.num_views = num_views

    def __len__(self):
        return len(self.voxels)

    def _apply_augs(self, voxel):
        if not self.augment:
            return voxel
            
        import random

        # 1. Random 90-degree rotations in the horizontal plane (assuming axes 0 and 2 are X and Z)
        k = random.randint(0, 3)
        if k > 0:
            voxel = torch.rot90(voxel, k, [0, 2])

        # 2. Block dropout
        if random.random() < self.aug_apply_prob:
            non_air_mask = voxel != 0
            drop_mask = (
                torch.rand_like(voxel, dtype=torch.float) < self.aug_dropout_prob
            )
            voxel[non_air_mask & drop_mask] = 0
            
        return voxel

    def __getitem__(self, idx):
        base_voxel = remap_voxel(
            self.voxels[idx], self.block_mapping, crop_bbox=self.crop_bbox
        )
        
        if self.num_views <= 1:
            return self._apply_augs(base_voxel.clone())
            
        return tuple(self._apply_augs(base_voxel.clone()) for _ in range(self.num_views))


class SchematicDataset(Dataset):
    """Dataset of (text, voxel, category) tuples for contrastive learning.

    Supports two parquet formats (auto-detected from config flags):
      - Numeric voxel IDs  (voxel_data column)       -> use_name_vocab=False
      - String block names (voxel_name_data column)  -> use_name_vocab=True

    Strategy flags:
      use_name_vocab       (Strategy 1): use string-name-based block vocabulary
      use_material_context (Strategy 3): append top-K block names to text
    """

    def __init__(
        self,
        df: pd.DataFrame,
        block_mapping: dict,
        crop_bbox: bool = True,
        augment: bool = False,
        aug_apply_prob: float = 0.5,
        aug_dropout_prob: float = 0.05,
        use_name_vocab: bool = False,
        use_material_context: bool = False,
        top_k_materials: int = 5,
        text_column: str = None,
        image_preprocess = None,
        num_views: int = 1,
    ):
        # Build text — priority: text_column > material_context > default
        if text_column and text_column in df.columns:
            self.texts = df[text_column].fillna("").tolist()
        elif use_material_context and "voxel_name_data" in df.columns:
            self.texts = [
                build_text_with_materials(row, top_k_materials=top_k_materials)
                for _, row in df.iterrows()
            ]
        else:
            self.texts = [build_text(row) for _, row in df.iterrows()]

        # Choose voxel source column and remap function (Strategy 1 or original)
        if use_name_vocab and "voxel_name_data" in df.columns:
            self.voxels = df["voxel_name_data"].tolist()
            self._remap_fn = remap_voxel_from_names
        else:
            self.voxels = df["voxel_data"].tolist()
            self._remap_fn = remap_voxel

        self.categories = df["subtitle"].fillna("Unknown").tolist()
        self.block_mapping = block_mapping
        self.crop_bbox = crop_bbox
        self.augment = augment
        self.aug_apply_prob = aug_apply_prob
        self.aug_dropout_prob = aug_dropout_prob
        
        self.image_preprocess = image_preprocess
        self.num_views = num_views
        self.image_paths_list = []
        
        for i in range(12):
            col = f"view_{i:02d}_path"
            if col in df.columns:
                self.image_paths_list.append(df[col].tolist())
                
        self.image_root = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        voxel = self._remap_fn(
            self.voxels[idx], self.block_mapping, crop_bbox=self.crop_bbox
        )
        if self.augment:
            voxel = augment_voxel(voxel, self.aug_apply_prob, self.aug_dropout_prob)
            
        if self.image_paths_list and self.image_preprocess is not None:
            import os
            import glob
            from PIL import Image
            
            total_available = len(self.image_paths_list)
            if self.num_views == 1:
                indices = [0]
            else:
                indices = np.round(np.linspace(0, total_available - 1, self.num_views)).astype(int)
                
            images_out = []
            for view_idx in indices:
                view_path = str(self.image_paths_list[view_idx][idx])
                
                if self.image_root is None:
                    parts = view_path.split("/")
                    pattern = f"**/*{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else f"**/{view_path}"
                    matches = glob.glob(pattern, recursive=True)
                    if matches:
                        matched_path = matches[0].replace('\\', '/')
                        self.image_root = matched_path[:matched_path.rfind(view_path)]
                    else:
                        self.image_root = ""
                        
                if self.image_root:
                    full_path = os.path.join(self.image_root, view_path)
                else:
                    full_path = view_path
                    
                try:
                    img = Image.open(full_path).convert('RGB')
                    images_out.append(self.image_preprocess(img))
                except Exception:
                    images_out.append(torch.zeros(3, 224, 224))
                    
            if self.num_views > 1:
                image_tensor = torch.stack(images_out)
            else:
                image_tensor = images_out[0]
                
            return text, voxel, image_tensor, self.categories[idx]
            
        return text, voxel, self.categories[idx]

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    cfg: dict,
    parquet_path: Optional[str] = None,
    image_preprocess = None,
):
    """Load data, preprocess, split, and return train/val/test DataLoaders.

    Strategy flags from cfg['data']:
      use_name_vocab       (Strategy 1): name-based block vocabulary
      use_material_context (Strategy 3): append block material names to text
      top_k_materials      : number of dominant materials to append

    Returns:
        train_loader, val_loader, test_loader, block_mapping, num_block_types, block_names

    Note: When use_name_vocab=True, block_mapping contains key
    '__index_to_name__' (list[str]) needed by Strategy 2 semantic embedding init.
    block_names is a list[str] of length num_block_types mapping compact_id → name.
    """
    data_cfg = cfg["data"]
    path = parquet_path or data_cfg["parquet_path"]

    use_name_vocab       = data_cfg.get("use_name_vocab",       False)
    use_material_context = data_cfg.get("use_material_context", False)
    top_k_materials      = data_cfg.get("top_k_materials",      5)

    # --- load (only columns actually needed to save RAM) --------------------
    import pyarrow.parquet as pq
    available_cols = set(pq.ParquetFile(path).schema_arrow.names)

    base_cols = {"subtitle", "title", "description", "tags"}
    
    use_trimodal = cfg.get("model", {}).get("use_trimodal", False)
    num_views = cfg.get("data", {}).get("num_views", 1)
    
    if use_trimodal:
        for i in range(12):
            col = f"view_{i:02d}_path"
            if col in available_cols:
                base_cols.add(col)

    if use_name_vocab and "voxel_name_data" in available_cols:
        base_cols.add("voxel_name_data")
    elif not use_name_vocab and "voxel_data" in available_cols:
        base_cols.add("voxel_data")

    text_mode = data_cfg.get("text_mode", "default")
    if text_mode in ("cleaned", "cleaned_aug") and "cleaned_text" in available_cols:
        base_cols.add("cleaned_text")
    if text_mode == "cleaned_aug" and "aug_1" in available_cols:
        base_cols.add("aug_1")

    load_cols = sorted(base_cols & available_cols)
    df = pd.read_parquet(path, columns=load_cols)
    print(f"Loaded {len(df)} samples from {path} (columns: {load_cols})")

    # Validate strategy requirements against available columns
    if use_name_vocab and "voxel_name_data" not in df.columns:
        print("  [WARN] use_name_vocab=True but 'voxel_name_data' not found. "
              "Falling back to numeric ID mapping.")
        use_name_vocab = False

    if use_material_context and "voxel_name_data" not in df.columns:
        print("  [WARN] use_material_context=True but 'voxel_name_data' not found. "
              "Disabling material context.")
        use_material_context = False

    # --- block mapping ------------------------------------------------------
    max_types = data_cfg["max_block_types"]

    if use_name_vocab:
        print("  [Strategy 1] Building name-based block vocabulary...")
        block_mapping = build_block_name_mapping(
            df["voxel_name_data"], max_types=max_types
        )
        n_unique = len([k for k in block_mapping
                        if k not in ("air", "__index_to_name__")])
        print(f"  Name vocab: {n_unique} unique non-air names (vocab={max_types})")
    else:
        block_mapping = build_block_mapping(
            df["voxel_data"], max_types=max_types
        )
        print(f"  Numeric ID vocab: {max_types} types")

    num_block_types = max_types

    # Build block_names list (compact_id → human-readable name)
    if use_name_vocab:
        block_names = block_mapping.get("__index_to_name__", ["unknown"] * num_block_types)
    else:
        block_names = extract_block_names(df, block_mapping)

    # --- splits -------------------------------------------------------------
    seed      = data_cfg["seed"]
    val_frac  = data_cfg["val_split"]
    test_frac = data_cfg["test_split"]

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=seed
    )
    relative_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=relative_test, random_state=seed
    )
    print(f"Splits — train: {len(idx_train)}, val: {len(idx_val)}, "
          f"test: {len(idx_test)}")

    crop_bbox        = data_cfg.get("crop_bbox",        True)
    augment          = data_cfg.get("augment",          True)
    aug_apply_prob   = data_cfg.get("aug_apply_prob",   0.5)
    aug_dropout_prob = data_cfg.get("aug_dropout_prob", 0.05)

    # resolve text column per split — aug_1 only for train to avoid leakage
    if text_mode == "cleaned":
        train_text_col = val_text_col = "cleaned_text"
    elif text_mode == "cleaned_aug":
        train_text_col = "aug_1"
        val_text_col   = "cleaned_text"
    else:
        train_text_col = val_text_col = None

    base_kwargs = dict(
        block_mapping        = block_mapping,
        crop_bbox            = crop_bbox,
        use_name_vocab       = use_name_vocab,
        use_material_context = use_material_context,
        top_k_materials      = top_k_materials,
        image_preprocess     = image_preprocess,
        num_views            = num_views if use_trimodal else 1,
    )

    ds_train = SchematicDataset(
        df.iloc[idx_train].reset_index(drop=True),
        augment=augment, aug_apply_prob=aug_apply_prob,
        aug_dropout_prob=aug_dropout_prob, text_column=train_text_col, **base_kwargs,
    )
    ds_val  = SchematicDataset(
        df.iloc[idx_val].reset_index(drop=True),
        augment=False, text_column=val_text_col, **base_kwargs,
    )
    ds_test = SchematicDataset(
        df.iloc[idx_test].reset_index(drop=True),
        augment=False, text_column=val_text_col, **base_kwargs,
    )

    # --- loaders ------------------------------------------------------------
    train_cfg = cfg["training"]

    def collate_fn(batch):
        if len(batch[0]) == 4:
            texts, voxels, images, categories = zip(*batch)
            return list(texts), torch.stack(voxels), torch.stack(images), list(categories)
        else:
            texts, voxels, categories = zip(*batch)
            return list(texts), torch.stack(voxels), list(categories)

    train_loader = DataLoader(
        ds_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
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
