# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands (run from repo root)

All scripts resolve paths from the repo root, not from `mc-retrieval/` (despite what README says — `mc-retrieval/src/` is a legacy stub with only `__pycache__/`).

```bash
# Bimodal training (text ↔ voxel)
python src/train.py --config configs/cnn/cnn_default.yaml
python src/train.py --config configs/pointbert/pb_s1s2_semantic_init.yaml
python src/train.py --config configs/pointbert/pb_s1s2_plan1.yaml --warmstart checkpoints/pb_s1s2/best.pt
python src/train.py --config configs/pointbert/pb_s1s2_semantic_init.yaml --resume checkpoints/pb_s1s2/last.pt

# Trimodal training (text + image + voxel) — same entrypoint, dispatched by model.trimodal_variant
python src/train.py --config configs/trimodal/pb_s1s2.yaml          # variant "pointbert" (HF CLIP ViT-L/14)
python src/precompute_clip.py --config configs/trimodal/trimodal_tinyclip.yaml  # required first for tinyclip
python src/train.py --config configs/trimodal/trimodal_tinyclip.yaml         # variant "tinyclip" (open_clip TinyCLIP)

# Evaluation
python src/evaluate.py --config configs/pointbert/pb_s1s2_semantic_init.yaml --checkpoint checkpoints/pb_s1s2/best.pt  # bimodal
python src/evaluate.py --config configs/trimodal/trimodal_tinyclip.yaml --checkpoint checkpoints/trimodal_tinyclip/best.pt   # tinyclip
python src/evaluate_trimodal.py --config configs/trimodal/pb_s1s2.yaml --checkpoint checkpoints/trimodal_pb_s1s2/best.pt  # pointbert

# CNN-only pretraining (MVM/SimCLR/hybrid)
python src/pretrain.py --config configs/cnn/cnn_default.yaml

# Baselines, demo
python src/baselines.py --config configs/cnn/cnn_default.yaml
python src/retrieval_demo.py --config configs/cnn/cnn_default.yaml --checkpoint checkpoints/best.pt
```

```bash
# Text data preprocessing (reformat/)
python reformat/clean_texts_openai.py    # raw metadata → cleaned_text (DeepSeek V4)
python reformat/augment_texts_openai.py  # cleaned_text → aug_1 paraphrases
```

## Architecture

### Bimodal (Text ↔ Voxel)

Both encoders project to a shared **256-dim L2-normalized** space with symmetric InfoNCE loss (`CLIPLoss` in `src/losses.py`).

**Voxel encoder** (`model.encoder_type` in config):
- `"cnn"` → `VoxelEncoder`: Embedding(block_id→32) → 3D Conv stack [64,128,256] → MLP(256)
- `"pointbert"` → `MinecraftPointBERTEncoder`: VoxelToPoints(32³→512 non-air pts) → Embedding(672,64) → input_proj(67→384) → 12-layer frozen Transformer → output_head(384→256)

**Text encoder**: frozen `all-MiniLM-L6-v2` (384-dim) + trainable linear projection.

### Trimodal (Text + Image + Voxel) — two coexisting variants

`src/train.py` dispatches between **two** trimodal schemes via `model.trimodal_variant`
(resolved by `resolve_variant()`; back-compat: `use_trimodal:true`→tinyclip,
`encoder_type:"trimodal"`→pointbert). Bimodal when the key is absent.

| `trimodal_variant` | Model | Image/Text backbone | Loss | Dataloader | Eval |
|---|---|---|---|---|---|
| `"pointbert"` (farhan) | `TriModalEncoder` (`model_trimodal.py`) | HF CLIP ViT-L/14 (768-d), separate `text_proj`/`image_proj`, `proj_type: linear\|mlp` | `TriModalCLIPLoss` (learnable τ, λ_TI/TV/IV) | `create_trimodal_dataloaders` (`dataset_trimodal.py`) | `evaluate_trimodal.py` |
| `"tinyclip"` (spunn) | `TrimodalEncoder` (`model.py`) | open_clip TinyCLIP (vendored `src/open_clip/`), shared `clip_proj` | `NTXentLoss` (TriCoLo, fixed τ, equal `/3`) | `create_dataloaders(image_preprocess=…)` (`dataset.py`) | `evaluate.py` |
| *absent* | `DualEncoder` | MiniLM text only | `CLIPLoss` | `create_dataloaders` | `evaluate.py` |

- **pointbert** views: evenly-spaced fixed indices via `_view_indices_for(12,6)` → `[0,2,4,7,9,11]`; per-view features cached `(N,6,768)` fp16, built automatically at dataset init (learnable `proj` stays outside cache).
- **tinyclip** views: 4, evenly-spaced; embeddings precomputed+averaged by **`precompute_clip.py`** (run before training when `use_cached_clip: true`). Also extends `pretrain.py` with MVM trimodal pretraining.
- Batch order differs per variant (handled in `train_one_epoch`/`validate`): pointbert `(texts,images,voxels)`, tinyclip `(texts,voxels,images)`.

### Semantic Strategies (Point-BERT only)

| Config flag | Strategy | Notes |
|---|---|---|
| `data.use_name_vocab: true` | S1 | Block vocab from string names; required for S2 |
| `model.use_semantic_init: true` | S2 | Init block embeddings from MiniLM of block name strings |
| `data.use_material_context: true` | S3 | Append top-K dominant block names to text query at train time |

S1/S2/S3 require `voxel_name_data` column in parquet — falls back silently with a warning if missing.

## Config Structure

```
configs/
  cnn/          # CNN voxel encoder (cnn_default.yaml, cnn_hybrid.yaml, cnn_simclr.yaml)
  pointbert/    # Point-BERT bimodal (pointbert.yaml, pb_s1s2_semantic_init.yaml, pb_s1s2s3_all.yaml, pointbert_finetune.yaml, ...)
  trimodal/     # Trimodal: pb_s1s2.yaml (variant "pointbert") + trimodal_tinyclip*.yaml (variant "tinyclip")
  ulip2/        # ULIP-2 pretrained Point-BERT variants (ulip2_*.yaml)
```

Folders are nested but filenames keep their original (pre-reorg) names, e.g. `configs/cnn/cnn_default.yaml`.

Key config keys: `model.encoder_type`, `model.embed_dim` (256), `pointbert.freeze_backbone`, `data.text_mode`, `data.crop_bbox`, `data.augment`.

## Data Files

Place in `data/` (git-ignored):

| File | Used by |
|---|---|
| `data.parquet` | CNN configs |
| `data_with_voxel_names.parquet` | Point-BERT S1/S2/S3 |
| `data_with_voxel_names_multiview_image.parquet` | Trimodal (also has `view_XX_path` columns) |
| `text_review.csv` / `text_review_cleaned_aug1.csv` | Text pipeline I/O |

`text_mode: "cleaned_aug"` uses `aug_1` column for training, `cleaned_text` for val/test.

## Checkpoints

`best.pt` / `last.pt` contain: `epoch`, `model_state`, `optimizer_state`, `criterion_state`, `val_loss`, `block_mapping`, `cfg`.

`block_mapping` maps raw block IDs → compact indices — must match between train and eval.

**Loading flags:**
- `--pretrained` — loads only `encoder_state` from pretrain.py output (CNN MVM pretraining only)
- `--warmstart` — loads full `model_state`, resets optimizer (Plan 2 → Plan 1 curriculum)
- `--resume` — loads full checkpoint including epoch/optimizer/scheduler

Point-BERT weights: download `Point-BERT.pth` from the [Point-BERT repo](https://github.com/lulutang0608/Point-BERT), place at `checkpoints/Point-BERT.pth`.

## Non-obvious Gotchas

- **`mc-retrieval/`** is a legacy mirror — actual source is root `src/`. Don't edit `mc-retrieval/src/`.
- **`fp_ir_4.py` and `untitled85 (1).py`** are Colab notebook exports — do not modify them.
- **Bbox crop is critical** — without `crop_bbox: true`, R@1 drops from 4.20% → 0.84%.
- **MVM pretraining only works with CNN encoder**, not Point-BERT.
- **FPS sampling** is non-deterministic at train time (random perm) but deterministic at eval when `use_fps_eval: true`. Point-BERT double-augmentation bug: `augment_voxel` (dataset.py) for CNN path, `augment_voxels` (train.py) for Point-BERT — don't enable both.
- **Trimodal image cache** is built once and stored as fp16 tensor; evenly-spaced fixed indices — not random per epoch. Path auto-derived from parquet filename.
- **`paper/` directory** contains the Typst research paper (`main.typ`). Text data preprocessing scripts live in `reformat/`.
- **Text cleaning/augmentation** uses DeepSeek V4 (`deepseek/deepseek-v4-flash`) via OpenAI-compatible API. Both scripts in `reformat/` use `ThreadPoolExecutor` + checkpoint-every-100-rows pattern.
- **No test suite, no lint config, no CI.** Development is Colab/GPU-based.
