# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Minecraft Schematic Retrieval research project with two components:
- **`mc-retrieval/`** — Main system: maps natural language queries and 3D voxel Minecraft structures into a shared semantic embedding space using a CLIP-style Dual Encoder. This is the active focus.
- **`cbir-thing/`** — Separate content-based image retrieval experiment (standalone).

## Commands

All scripts are run from within `mc-retrieval/`. Install dependencies first:

```bash
pip install -r mc-retrieval/requirements.txt
```

### Training
```bash
# CNN-based baseline
python src/train.py --config configs/default.yaml

# Point-BERT (frozen backbone — recommended starting point)
python src/train_pointbert.py --config configs/pointbert.yaml

# Point-BERT with semantic strategies
python src/train_pointbert.py --config configs/pb_s1s2_semantic_init.yaml
python src/train_pointbert.py --config configs/pb_s1s2s3_all.yaml

# Warm-start Point-BERT fine-tune from a Plan 2 checkpoint
python src/train_pointbert.py --config configs/pointbert_finetune.yaml \
    --warmstart checkpoints/pointbert_plan2/best.pt
```

### Evaluation
```bash
python src/evaluate.py --config configs/pb_s1s2s3_all.yaml \
    --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```

### Baselines
```bash
python src/baselines.py --config configs/default.yaml
```

### Utilities
```bash
# Export text inputs to CSV for review
python src/export_texts.py --parquet data/data_with_voxel_names_multiview_image.parquet
python src/export_texts.py --config configs/pb_s1s2_semantic_init.yaml

# Interactive retrieval demo
python src/retrieval_demo.py --config configs/pb_s1s2s3_all.yaml \
    --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```

## Architecture

### Dual-Encoder (CLIP-style)
Both encoders project into a shared **256-dim L2-normalized** space. Training uses symmetric InfoNCE (CLIPLoss).

**Text Encoder**: `all-MiniLM-L6-v2` (frozen, 22M params) → Linear projection head (trainable, 384→256→256).

**Voxel Encoder (Point-BERT path)**:
1. `VoxelToPoints`: 32³ grid → 512 sparse non-air points (FPS during eval, random during train)
2. `nn.Embedding(672, 64)`: block ID → 64-dim features (trainable; optionally semantic-initialized)
3. Concat xyz(3) + block_feats(64) → `input_proj` Linear(67→384) (trainable)
4. 12-layer frozen Point-BERT Transformer (384-dim, 6-head; pretrained on ShapeNet55)
5. `output_head` Linear(384→256) (trainable)

Total trainable params: ~653K out of ~44M total.

**Voxel Encoder (CNN path)**: `model.py` — block embedding → 3D Conv stack → projection (full alternative to Point-BERT).

### Data
- Two parquet formats: `data.parquet` (numeric block IDs) and `data_with_voxel_names_multiview_image.parquet` (string block names, needed for Strategies 1–3).
- 8,328 samples, split 80/10/10 (train/val/test).
- `dataset.py` handles both formats; the `use_name_vocab` / `use_material_context` flags in config select the strategy.

### Semantic Enhancement Strategies (Point-BERT only)
| Flag in config | Strategy | Effect |
|---|---|---|
| `use_name_vocab: true` | S1 | Vocabulary built from block names instead of IDs |
| `use_semantic_init: true` | S2 | Block embeddings initialized from MiniLM encodings; cached under `checkpoints/cache/block_emb_{md5}.npy` |
| `use_material_context: true` | S3 | Top-K dominant block names appended to text query |

### Key Config Fields
Configs live in `mc-retrieval/configs/`. Important fields:
- `data.parquet_path` — path to dataset file
- `pointbert.pretrained_path` — path to `Point-BERT.pth` (download from Point-BERT GitHub)
- `pointbert.freeze_backbone` — `true` for Plan 2 (adapter-only), `false` for full fine-tune
- `training.checkpoint_dir` — where checkpoints are saved

### Metrics
Evaluation reports **Recall@1/5/10** and **MRR** for text→voxel and voxel→text retrieval, plus category-level breakdown.
