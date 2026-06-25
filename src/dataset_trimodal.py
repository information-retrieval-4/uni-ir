"""Dataset for trimodal (text + image + voxel) retrieval."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from sklearn.model_selection import train_test_split

from dataset import (
    build_block_name_mapping,
    remap_voxel_from_names,
    augment_voxel,
    build_text,
    build_text_with_materials,
)


def resolve_image_path(
    row: pd.Series,
    renders_base: str,
    view_idx: int,
    n_views: int = 12,
) -> Optional[str]:
    """Resolve rendered image path from a parquet row.

    Priority:
    1. view_XX_path column (absolute or relative via renders_base)
    2. render_folder column + view_XX.jpg filename
    """
    col = f"view_{view_idx:02d}_path"
    stored = row.get(col)
    if isinstance(stored, str) and stored:
        if os.path.exists(stored):
            return stored
        parts = Path(stored).parts
        if len(parts) >= 2:
            candidate = os.path.join(renders_base, parts[-2], parts[-1])
            if os.path.exists(candidate):
                return candidate

    render_folder = row.get("render_folder")
    if isinstance(render_folder, str):
        candidate = os.path.join(renders_base, render_folder, f"view_{view_idx:02d}.jpg")
        if os.path.exists(candidate):
            return candidate

    return None


def _view_indices_for(n_views: int, n_views_use: int) -> list:
    """Evenly-spaced view indices from [0, n_views), length = n_views_use."""
    if n_views_use <= 1:
        return [n_views // 2]
    return [round(i * (n_views - 1) / (n_views_use - 1)) for i in range(n_views_use)]


def _build_image_cache(df: pd.DataFrame, cfg: dict, processor: CLIPProcessor, cache_path: str):
    """Pre-extract frozen CLIP vision features for all samples × all views.

    Saves (N, N_views_use, clip_out_dim) float16 tensor to cache_path.
    Only the frozen CLIP backbone + visual_projection are used here;
    the learnable self.proj layer remains outside the cache.
    """
    from transformers import CLIPModel
    from tqdm import tqdm

    data_cfg = cfg["data"]
    clip_name = cfg["model"].get("clip_model", "openai/clip-vit-base-patch16")
    renders_base = data_cfg.get("renders_base", "data/renders")
    n_views = data_cfg.get("n_views", 12)
    n_views_use = data_cfg.get("n_views_use", 6)
    image_size = cfg["model"].get("image_size", 224)
    view_indices = _view_indices_for(n_views, n_views_use)
    batch_imgs = 32  # images per GPU batch during cache build

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip = CLIPModel.from_pretrained(clip_name)
    vision_model = clip.vision_model.to(device).eval()
    visual_proj = clip.visual_projection.to(device).eval()
    del clip

    print(f"[Cache] Building image cache for {len(df)} samples × {n_views_use} views "
          f"on {device} ...")

    fallback = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    N = len(df)

    # Flatten all (sample × view) images into one list, then batch-encode
    all_imgs = []
    for i in range(N):
        row = df.iloc[i]
        for vi in view_indices:
            img_path = resolve_image_path(row, renders_base, vi, n_views)
            try:
                img = Image.open(img_path).convert("RGB") if img_path else fallback
            except Exception:
                img = fallback
            all_imgs.append(img)

    total = len(all_imgs)
    all_feats = []
    with torch.no_grad():
        for start in tqdm(range(0, total, batch_imgs), desc="[Cache] CLIP encode"):
            batch = all_imgs[start:start + batch_imgs]
            pv = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            out = vision_model(pixel_values=pv)
            feats = visual_proj(out.pooler_output).cpu().half()  # (b, clip_out_dim) fp16
            all_feats.append(feats)

    del vision_model, visual_proj, all_imgs
    cache = torch.cat(all_feats, dim=0).view(N, n_views_use, -1)  # (N, V, D)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[Cache] Saved {cache.shape} fp16 → {cache_path} "
          f"({cache.numel() * 2 / 1e6:.1f} MB)")
    return cache


class TriModalDataset(Dataset):
    """Each item: (text, image_feats_or_pixels, voxel, category).

    When image_cache is provided:
      image output shape: (N_views, clip_out_dim)  — pre-extracted CLIP features
    Otherwise:
      image output shape: (N_views, 3, H, W)  — raw pixel values
    """

    def __init__(
        self,
        df: pd.DataFrame,
        block_mapping: dict,
        processor: CLIPProcessor,
        cfg: dict,
        split: str = "train",
        image_cache: Optional[torch.Tensor] = None,
    ):
        data_cfg = cfg["data"]
        self.df = df.reset_index(drop=True)
        self.block_mapping = block_mapping
        self.processor = processor
        self.renders_base = data_cfg.get("renders_base", "data/renders")
        self.n_views = data_cfg.get("n_views", 12)
        self.n_views_use = data_cfg.get("n_views_use", 6)
        self.image_size = cfg["model"].get("image_size", 224)
        self.is_train = split == "train"
        self.crop_bbox = data_cfg.get("crop_bbox", True)
        self.aug = data_cfg.get("augment", False) and self.is_train
        self.aug_apply_prob = data_cfg.get("aug_apply_prob", 0.5)
        self.aug_dropout_prob = data_cfg.get("aug_dropout_prob", 0.05)
        self.fallback_img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        self._view_indices = _view_indices_for(self.n_views, self.n_views_use)

        # image_cache: (N_subset, N_views, clip_out_dim) fp16 or None
        self.image_cache = image_cache

        use_material_context = data_cfg.get("use_material_context", False)
        if use_material_context and "voxel_name_data" in df.columns:
            top_k = data_cfg.get("top_k_materials", 5)
            self.texts = [
                build_text_with_materials(df.iloc[i], top_k_materials=top_k)
                for i in range(len(df))
            ]
        else:
            self.texts = [build_text(df.iloc[i]) for i in range(len(df))]

        self.voxel_name_data = df["voxel_name_data"].tolist()
        self.categories = df["subtitle"].fillna("Unknown").tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = self.texts[idx]

        if self.image_cache is not None:
            # (N_views, clip_out_dim) fp16 → float32 for model
            pixel_values = self.image_cache[idx].float()
        else:
            # load raw images → (N_views, 3, H, W)
            views = []
            for vi in self._view_indices:
                img_path = resolve_image_path(row, self.renders_base, vi, self.n_views)
                try:
                    img = Image.open(img_path).convert("RGB") if img_path else self.fallback_img
                except Exception:
                    img = self.fallback_img
                pv = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
                views.append(pv)
            pixel_values = torch.stack(views, dim=0)

        voxel = remap_voxel_from_names(
            self.voxel_name_data[idx], self.block_mapping, crop_bbox=self.crop_bbox
        )
        if self.aug:
            voxel = augment_voxel(voxel, self.aug_apply_prob, self.aug_dropout_prob)

        return text, pixel_values, voxel, self.categories[idx]


def _collate_trimodal(batch):
    texts, images, voxels, categories = zip(*batch)
    return list(texts), torch.stack(images), torch.stack(voxels), list(categories)


def create_trimodal_dataloaders(cfg: dict):
    """Load trimodal parquet, split train/val/test, return DataLoaders + metadata.

    If cfg['data']['image_cache_path'] is set:
      - Build cache on first run (one-time ~2-3 min), then load from disk.
      - Training skips CLIP vision backbone entirely → ~10x faster per epoch.

    Returns:
        train_loader, val_loader, test_loader, block_mapping, num_block_types, processor
    """
    data_cfg = cfg["data"]
    path = data_cfg["parquet_path"]
    clip_name = cfg["model"].get("clip_model", "openai/clip-vit-base-patch16")

    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(path).schema_arrow.names)
    needed = {"subtitle", "title", "description", "tags", "voxel_name_data"}
    n_views = data_cfg.get("n_views", 12)
    for i in range(n_views):
        col = f"view_{i:02d}_path"
        if col in available:
            needed.add(col)
    if "render_folder" in available:
        needed.add("render_folder")

    load_cols = sorted(needed & available)
    df = pd.read_parquet(path, columns=load_cols)
    print(f"[Trimodal] Loaded {len(df)} samples | columns: {load_cols}")

    max_types = data_cfg["max_block_types"]
    block_mapping = build_block_name_mapping(df["voxel_name_data"], max_types=max_types)
    num_block_types = max_types
    print(f"[Trimodal] Block vocab: {max_types} types")

    processor = CLIPProcessor.from_pretrained(clip_name)

    # --- image cache (optional) ---
    cache_path = data_cfg.get("image_cache_path")
    image_cache = None
    if cache_path:
        if not os.path.exists(cache_path):
            image_cache = _build_image_cache(df, cfg, processor, cache_path)
        else:
            image_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            print(f"[Cache] Loaded image cache {image_cache.shape} from {cache_path}")

        # sanity check: cache must match current config
        n_views_use = data_cfg.get("n_views_use", 6)
        if image_cache.shape[0] != len(df) or image_cache.shape[1] != n_views_use:
            print(f"[Cache] WARNING: cache shape {image_cache.shape} doesn't match "
                  f"({len(df)}, {n_views_use}, *). Rebuilding ...")
            image_cache = _build_image_cache(df, cfg, processor, cache_path)

    # --- split ---
    val_split = data_cfg.get("val_split", 0.1)
    test_split = data_cfg.get("test_split", 0.1)
    seed = data_cfg.get("seed", 42)

    labels = df["subtitle"].fillna("Unknown").tolist()
    idx_all = list(range(len(df)))
    idx_train, idx_temp, _, labels_temp = train_test_split(
        idx_all, labels,
        test_size=val_split + test_split,
        stratify=labels,
        random_state=seed,
    )
    val_frac = val_split / (val_split + test_split)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=1 - val_frac,
        stratify=labels_temp,
        random_state=seed,
    )
    print(f"[Trimodal] Split — train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    num_workers = cfg["training"].get("num_workers", 2)
    batch_size = cfg["training"]["batch_size"]

    def make_loader(indices, split):
        cache_subset = image_cache[indices] if image_cache is not None else None
        ds = TriModalDataset(
            df.iloc[indices], block_mapping, processor, cfg,
            split=split, image_cache=cache_subset,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=_collate_trimodal,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return (
        make_loader(idx_train, "train"),
        make_loader(idx_val, "val"),
        make_loader(idx_test, "test"),
        block_mapping,
        num_block_types,
        processor,
    )
