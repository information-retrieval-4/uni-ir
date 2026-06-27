# Configs

All configs are organized by schema. Pass any config to the unified entrypoint:

```bash
python src/train.py --config configs/<folder>/<file>.yaml
```

---

## `cnn/` — CNN Bimodal (Text + Voxel)

Simplest baseline. No large GPU required.

**Architecture:** `VoxelEncoder` (3D Conv `[64,128,256]`) + `all-MiniLM-L6-v2` (frozen) → `CLIPLoss`

| Config | Notes |
|---|---|
| `cnn_default.yaml` | Main baseline — start here |
| `cnn_simclr.yaml` | + SimCLR voxel pretraining mode |
| `cnn_hybrid.yaml` | + Hybrid pretraining (MVM + SimCLR) |

```bash
python src/train.py --config configs/cnn/cnn_default.yaml
python src/evaluate.py --config configs/cnn/cnn_default.yaml --checkpoint checkpoints/best.pt
```

---

## `pointbert/` — PointBERT Bimodal (Text + PointBERT Voxel)

Replaces the CNN voxel encoder with a frozen pretrained ViT-style Transformer.
Requires `checkpoints/Point-BERT.pth` ([download](https://github.com/lulutang0608/Point-BERT)).

**Architecture:** `MinecraftPointBERTEncoder` (12L, 384-dim) + `all-MiniLM-L6-v2` (frozen) → `CLIPLoss`

**Semantic strategies** layered on top of the backbone:

| Strategy | Config key | Effect |
|---|---|---|
| S1 | `use_name_vocab: true` | Block vocab from string names (required for S2) |
| S2 | `use_semantic_init: true` | Init block embeddings from SentenceTransformer of block names |
| S3 | `use_material_context: true` | Append top-K dominant block names to text query at train time |

| Config | Strategies | Backbone | Notes |
|---|---|---|---|
| `pointbert.yaml` | — | Frozen | Vanilla, no strategies |
| `pointbert_finetune.yaml` | — | Unfrozen | Full fine-tune |
| `pb_s1_name_vocab.yaml` | S1 | Frozen | |
| `pb_s1s2_semantic_init.yaml` | S1+S2 | Frozen | **Recommended starting point** |
| `pb_s1s2_plan1.yaml` | S1+S2 | Unfrozen | Phase 2 after `pb_s1s2_semantic_init` |
| `pb_s3_material_ctx.yaml` | S3 | Frozen | Experimental |
| `pb_s1s2s3_all.yaml` | S1+S2+S3 | Frozen | All strategies |

```bash
# Phase 1 — frozen backbone
python src/train.py --config configs/pointbert/pb_s1s2_semantic_init.yaml

# Phase 2 — unfreeze backbone
python src/train.py --config configs/pointbert/pb_s1s2_plan1.yaml \
    --warmstart checkpoints/pb_s1s2_semantic_init_512/best.pt

python src/evaluate.py --config configs/pointbert/pb_s1s2_semantic_init.yaml \
    --checkpoint checkpoints/pb_s1s2_semantic_init_512/best.pt
```

---

## `ulip2/` — ULIP-2 PointBERT Bimodal

Same as PointBERT bimodal but the backbone is initialized from **ULIP-2** — pretrained on Objaverse + ShapeNet with 3D+image+text alignment, so it starts with richer representations.

Requires a ULIP-2 checkpoint from [ULIP GitHub → Model Zoo](https://github.com/salesforce/ULIP).

| Config | Points | `block_embed_dim` | Notes |
|---|---|---|---|
| `ulip2_pointbert_10k.yaml` | 1024 | 128 | 10K colored pts — **recommended** |
| `ulip2_pointbert_8k.yaml` | 1024 | 128 | 8K colored pts |
| `ulip2_pointnext.yaml` | — | — | PointNeXt architecture variant |
| `ulip2_pointbert_10k_plan1.yaml` | 1024 | 256 | Phase 2: unfreeze backbone |
| `ulip2_pointbert_8k_plan1.yaml` | 1024 | 128 | Phase 2: unfreeze backbone |

```bash
python src/train.py --config configs/ulip2/ulip2_pointbert_10k.yaml

python src/train.py --config configs/ulip2/ulip2_pointbert_10k_plan1.yaml \
    --warmstart checkpoints/ulip2_pointbert_10k/best.pt
```

---

## `trimodal/` — Trimodal (Text + Image + Voxel)

Two coexisting trimodal schemes, selected by `model.trimodal_variant` in the config.

| | `"pointbert"` | `"tinyclip"` |
|---|---|---|
| Voxel | PointBERT | CNN `[128,256,512]` |
| Text + Image | HF CLIP ViT-L/14 (768-dim), separate projections | open_clip TinyCLIP-45M, shared projection |
| Loss | `TriModalCLIPLoss` — learnable τ, configurable λ | `NTXentLoss` — fixed τ=0.07, equal `/3` |
| Image cache | Auto-built `(N,6,768)` fp16, per-view | Pre-averaged via `precompute_clip.py` (manual step) |
| Views | 6 of 12 (fixed, evenly spaced) | 4 views, averaged |
| Eval | `src/evaluate_trimodal.py` | `src/evaluate.py` |
| Data | `data_with_voxel_names_multiview_image.parquet` | `data_merged_with_text.parquet` |

| Config | Variant | Notes |
|---|---|---|
| `pb_s1s2.yaml` | `pointbert` | S1+S2, frozen backbone — start here for pointbert trimodal |
| `trimodal_tinyclip.yaml` | `tinyclip` | Default tinyclip config |
| `trimodal_tinyclip_hybrid.yaml` | `tinyclip` | + Hybrid pretraining (MVM + SimCLR) |

```bash
# pointbert variant
python src/train.py --config configs/trimodal/pb_s1s2.yaml
python src/evaluate_trimodal.py --config configs/trimodal/pb_s1s2.yaml \
    --checkpoint checkpoints/trimodal_pb_s1s2/best.pt

# tinyclip variant — precompute embeddings first (one-time)
python src/precompute_clip.py --config configs/trimodal/trimodal_tinyclip.yaml
python src/train.py --config configs/trimodal/trimodal_tinyclip.yaml
python src/evaluate.py --config configs/trimodal/trimodal_tinyclip.yaml \
    --checkpoint checkpoints/trimodal_tinyclip/best.pt
```
