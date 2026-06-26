# Minecraft Schematic Retrieval (`mc-retrieval`)

Cross-modal retrieval system that maps natural language queries and 3D voxel Minecraft structures into a shared semantic embedding space using a CLIP-style Dual Encoder.

---

## Repository Structure

```text
mc-retrieval/
├── configs/                   # YAML configs for all experiments
│   ├── default.yaml           # CNN baseline
│   ├── pointbert.yaml         # Point-BERT frozen backbone
│   ├── pb_s1s2_semantic_init.yaml   # Point-BERT + S1 + S2
│   ├── pb_s1s2s3_all.yaml           # Point-BERT + S1 + S2 + S3
│   └── pointbert_finetune.yaml      # Point-BERT full fine-tune
├── data/                      # Parquet dataset files (not committed)
├── src/
│   ├── dataset.py             # Preprocessing, dataloaders, augmentation
│   ├── model.py               # Unified dual encoder (CNN & Point-BERT)
│   ├── train.py               # Unified training loop
│   ├── pretrain.py            # Masked Voxel Modeling (MVM) pretraining
│   ├── evaluate.py            # Recall@K, MRR evaluation (auto-detects backend)
│   ├── baselines.py           # TF-IDF, Random, unimodal baselines
│   ├── retrieval_demo.py      # Interactive text→voxel retrieval demo
│   ├── export_texts.py        # Export text inputs to CSV for inspection
│   ├── visualize.py           # Result visualizations
│   └── utils.py               # Shared utilities
├── requirements.txt
└── README.md                  # This file
```

---

## Getting Started

### Installation

```bash
pip install -r mc-retrieval/requirements.txt
```

### Data

Place dataset parquet files in `mc-retrieval/data/`:

- `data.parquet` — numeric block IDs (CNN path)
- `data_with_voxel_names_multiview_image.parquet` — string block names (Point-BERT S1/S2/S3)

### Point-BERT Weights

Download `Point-BERT.pth` from the [Point-BERT GitHub](https://github.com/lulutang0608/Point-BERT) and place it at `mc-retrieval/checkpoints/Point-BERT.pth`.

---

## Training

All commands run from `mc-retrieval/`.

### CNN path

```bash
# Plain CNN baseline
python src/train.py --config configs/cnn/cnn_default.yaml

# With MVM pretrained voxel encoder
python src/train.py --config configs/cnn/cnn_default.yaml \
    --pretrained checkpoints/pretrained_voxel.pt
```

### Point-BERT path

Point-BERT configuration is toggled by setting `encoder_type: "pointbert"` in your config file.

```bash
# Frozen backbone (recommended starting point)
python src/train.py --config configs/pointbert/pointbert.yaml

# With semantic strategies S1+S2
python src/train.py --config configs/pointbert/pb_s1s2_semantic_init.yaml

# With all strategies S1+S2+S3
python src/train.py --config configs/pointbert/pb_s1s2s3_all.yaml

# Full fine-tune warm-started from Plan 2 checkpoint
python src/train.py --config configs/pointbert/pointbert_finetune.yaml \
    --warmstart checkpoints/pointbert_plan2/best.pt
```

### MVM Pretraining (CNN only)

```bash
python src/pretrain.py --config configs/cnn/cnn_default.yaml
```

---

## Evaluation

```bash
# evaluate.py auto-detects CNN vs Point-BERT from config
python src/evaluate.py --config configs/pointbert/pb_s1s2s3_all.yaml \
    --checkpoint checkpoints/pb_s1s2s3_all/best_model.pth
```

Reports **Recall@1/5/10** and **MRR** for text→voxel and voxel→text, plus category-level breakdown.

---

## Architecture

Both encoders project into a shared **256-dim L2-normalized** space. Loss: symmetric InfoNCE (CLIPLoss).

### Text Encoder

`all-MiniLM-L6-v2` (frozen, 22M params) → Linear projection head (trainable, 384→256→256).

### Voxel Encoder — CNN config (`encoder_type: "cnn"`)

`Embedding(256, 32)` → 3D Conv stack `[64, 128, 256]` → MLP(256). Optional MVM pretraining warm-start.

### Voxel Encoder — Point-BERT config (`encoder_type: "pointbert"`)

1. `VoxelToPoints`: 32³ grid → 512 sparse non-air points
2. `nn.Embedding(672, 64)`: block ID → 64-dim features
3. Concat xyz(3) + block_feats(64) → `input_proj` Linear(67→384)
4. 12-layer frozen Point-BERT Transformer (384-dim, pretrained on ShapeNet55)
5. `output_head` Linear(384→256)

Trainable params: ~653K / 44M total.

---

## Semantic Enhancement Strategies (Point-BERT only)

| Config flag                  | Strategy | Effect                                                             |
| ---------------------------- | -------- | ------------------------------------------------------------------ |
| `use_name_vocab: true`       | S1       | Vocabulary from block name strings instead of opaque numeric IDs   |
| `use_semantic_init: true`    | S2       | Block embeddings initialized from MiniLM encodings of block names  |
| `use_material_context: true` | S3       | Top-K dominant block names appended to text query at training time |

---

## Key Results (CNN Path)

Best config: **+Bbox Crop + MVM Pretrain + Augmentation**

| Metric          | Baseline | +Crop+Pretrain+Aug |
| --------------- | -------- | ------------------ |
| Text→Voxel R@1  | 0.84%    | **4.20%**          |
| Text→Voxel R@10 | 17.65%   | **25.21%**         |
| Text→Voxel MRR  | 0.0632   | **0.1105**         |
| Cat P@1 (T→V)   | 39.62%   | **51.62%**         |

See [`idea-explainer/full_results.md`](idea-explainer/full_results.md) for full ablation tables.
