# Ablation Results — Full Metrics

## Instance-Level Retrieval

### Text → Voxel

| Metric | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain |
|--------|----------|-----------|------------|----------------|
| Recall@1 | 0.84% | 1.32% | 3.72% | **3.96%** |
| Recall@5 | 8.64% | 8.52% | 13.69% | **14.89%** |
| Recall@10 | 17.65% | 15.85% | 21.25% | **24.13%** |
| MRR | 0.0632 | 0.0688 | 0.0985 | **0.1080** |
| Median Rank | 55 | 52 | 41 | **36** |
| Mean Rank | 91.23 | 89.79 | 83.85 | **76.61** |

### Voxel → Text

| Metric | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain |
|--------|----------|-----------|------------|----------------|
| Recall@1 | 2.16% | 2.28% | 2.52% | **3.36%** |
| Recall@5 | 8.16% | 8.88% | 11.40% | **13.93%** |
| Recall@10 | 16.33% | 15.25% | 20.65% | **23.53%** |
| MRR | 0.0682 | 0.0722 | 0.0863 | **0.1016** |
| Median Rank | 57 | 53 | 42 | **36** |
| Mean Rank | 94.09 | 91.36 | 85.27 | **77.75** |

---

## Category-Level Retrieval

### Text → Voxel

| Metric | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain |
|--------|----------|-----------|------------|----------------|
| Cat P@1 | 39.62% | 42.62% | 47.90% | **50.30%** |
| Cat P@5 | 42.64% | 43.77% | 46.46% | **47.88%** |
| Cat P@10 | 43.51% | 43.33% | 46.22% | **46.61%** |
| Cat Hit@1 | 39.62% | 42.62% | 47.90% | **50.30%** |
| Cat Hit@5 | 82.23% | 82.35% | **84.03%** | 83.31% |
| Cat Hit@10 | **91.72%** | 89.92% | 92.08% | 91.72% |

### Voxel → Text

| Metric | Baseline | +Pretrain | +Bbox Crop | +Crop+Pretrain |
|--------|----------|-----------|------------|----------------|
| Cat P@1 | 45.50% | 45.50% | 46.70% | **48.62%** |
| Cat P@5 | 44.95% | 45.40% | 46.51% | **48.04%** |
| Cat P@10 | 44.11% | 44.56% | 45.89% | **46.87%** |
| Cat Hit@1 | 45.50% | 45.50% | 46.70% | **48.62%** |
| Cat Hit@5 | 74.55% | 74.07% | 74.67% | **76.59%** |
| Cat Hit@10 | 82.35% | 83.67% | 82.71% | **84.39%** |

---

## Training Details

| Config | Val Loss | Best Epoch | Total Epochs |
|--------|----------|------------|--------------|
| Baseline | 3.9112 | 88 | 100 |
| +Pretrain | 3.8535 | 96 | 100 |
| +Bbox Crop | 3.7638 | 77 | 92 (early stop) |
| +Crop+Pretrain | — | — | — |

## Key Findings

1. **Bbox cropping is the single most impactful change** — 4.4× improvement in T→V R@1, driven by 12.7× increase in input density (3% → 38% non-air voxels)
2. **MVM pretraining is additive when combined with cropping** — marginal alone (+0.48 R@1), but meaningful on top of crop (+2.88 R@10)
3. **Category-level P@1 broke 50%** in the best config — majority of top-1 retrievals match the correct category
4. **Median rank dropped from 55 → 36** (top 4.3% of gallery) in the best config
