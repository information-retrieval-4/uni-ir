# Full Results — MC Schematic Cross-Modal Retrieval

## 1. Cross-Modal Ablation (Our Model)

### Instance-Level: Text → Voxel

| Metric      | Baseline | +MVM Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Crop+Pretrain+Aug+Semantic Init |
| ----------- | -------- | ------------- | ---------- | -------------- | ------------------ | -------------- |
| Recall@1    | 0.84%    | 1.32%         | 3.72%      | 3.96%          | 4.20%              | **4.44%**      |
| Recall@5    | 8.64%    | 8.52%         | 13.69%     | **14.89%**     | 14.77%             | 14.77%         |
| Recall@10   | 17.65%   | 15.85%        | 21.25%     | 24.13%         | **25.21%**         | 23.41%         |
| MRR         | 0.0632   | 0.0688        | 0.0985     | 0.1080         | 0.1105             | **0.1132**     |
| Median Rank | 55       | 52            | 41         | 36             | **33**             | 35             |
| Mean Rank   | 91.23    | 89.79         | 83.85      | 76.61          | **73.80**          | 75.04          |

### Instance-Level: Voxel → Text

| Metric      | Baseline | +MVM Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Crop+Pretrain+Aug+Semantic Init |
| ----------- | -------- | ------------- | ---------- | -------------- | ------------------ | -------------- |
| Recall@1    | 2.16%    | 2.28%         | 2.52%      | 3.36%          | 3.36%              | **4.20%**      |
| Recall@5    | 8.16%    | 8.88%         | 11.40%     | 13.93%         | **14.05%**         | 13.69%         |
| Recall@10   | 16.33%   | 15.25%        | 20.65%     | 23.53%         | **23.77%**         | 22.93%         |
| MRR         | 0.0682   | 0.0722        | 0.0863     | 0.1016         | 0.1019             | **0.1062**     |
| Median Rank | 57       | 53            | 42         | 36             | **34**             | 36             |
| Mean Rank   | 94.09    | 91.36         | 85.27      | 77.75          | **75.78**          | 76.81          |

---

### Category-Level: Text → Voxel

| Metric     | Baseline | +MVM Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Crop+Pretrain+Aug+Semantic Init |
| ---------- | -------- | ------------- | ---------- | -------------- | ------------------ | -------------- |
| Cat P@1    | 39.62%   | 42.62%        | 47.90%     | 50.30%         | **51.62%**         | 46.58%         |
| Cat P@5    | 42.64%   | 43.77%        | 46.46%     | **47.88%**     | 47.37%             | 46.46%         |
| Cat P@10   | 43.51%   | 43.33%        | 46.22%     | 46.61%         | **46.81%**         | 45.86%         |
| Cat Hit@1  | 39.62%   | 42.62%        | 47.90%     | 50.30%         | **51.62%**         | 46.58%         |
| Cat Hit@5  | 82.23%   | 82.35%        | 84.03%     | 83.31%         | 82.47%             | **84.27%**     |
| Cat Hit@10 | 91.72%   | 89.92%        | 92.08%     | 91.72%         | 91.72%             | **92.32%**     |

### Category-Level: Voxel → Text

| Metric     | Baseline | +MVM Pretrain | +Bbox Crop | +Crop+Pretrain | +Crop+Pretrain+Aug | +Crop+Pretrain+Aug+Semantic Init |
| ---------- | -------- | ------------- | ---------- | -------------- | ------------------ | -------------- |
| Cat P@1    | 45.50%   | 45.50%        | 46.70%     | **48.62%**     | 46.82%             | 46.46%         |
| Cat P@5    | 44.95%   | 45.40%        | 46.51%     | **48.04%**     | 47.42%             | 46.27%         |
| Cat P@10   | 44.11%   | 44.56%        | 45.89%     | **46.87%**     | 46.73%             | 45.40%         |
| Cat Hit@1  | 45.50%   | 45.50%        | 46.70%     | **48.62%**     | 46.82%             | 46.46%         |
| Cat Hit@5  | 74.55%   | 74.07%        | 74.67%     | 76.59%         | 76.35%             | **78.27%**     |
| Cat Hit@10 | 82.35%   | 83.67%        | 82.71%     | 84.39%         | 83.43%             | **85.71%**     |

---

## 2. Unimodal Baselines (Category-Level Only)

> [!NOTE]
> Instance-level metrics (recall, MRR, rank) are N/A for unimodal baselines — there is no cross-modal ground-truth pair. Category-level metrics compare against same-modality retrieval.

| Metric     | Random | BM25 (TF-IDF) | Text-Only (ST) | Voxel-Only | Ours   |
| ---------- | ------ | ------------- | -------------- | ---------- | ------ |
| Cat P@1    | ~31%\* | 70.59%        | **78.27%**     | 41.78%     | 51.62% |
| Cat P@5    | ~31%\* | 69.77%        | **74.84%**     | 41.10%     | 47.88% |
| Cat P@10   | ~31%\* | 68.98%        | **72.23%**     | 40.92%     | 46.81% |
| Cat Hit@1  | ~31%\* | 70.59%        | **78.27%**     | 41.78%     | 51.62% |
| Cat Hit@5  | ~31%\* | **94.12%**    | 93.04%         | 76.35%     | 84.03% |
| Cat Hit@10 | ~31%\* | 96.64%        | **96.76%**     | 85.23%     | 91.72% |

\* Random cat P@1 approximated by largest category proportion (259/833 ≈ 31%)

### Random Baseline (Analytical)

| Metric      | Value  |
| ----------- | ------ |
| Recall@1    | 0.12%  |
| Recall@5    | 0.60%  |
| Recall@10   | 1.20%  |
| MRR         | 0.0088 |
| Median Rank | 416.5  |
| Mean Rank   | 417.0  |

---

## 3. MVM Pretraining Probe

| Mask Ratio | Loss   | Overall Accuracy | Non-Air Accuracy |
| ---------- | ------ | ---------------- | ---------------- |
| 10%        | 0.0192 | 99.54%           | 99.54%           |
| 20%        | 0.0292 | 99.29%           | 99.29%           |
| 30%        | 0.0468 | 98.89%           | 98.89%           |
| 50%        | 0.1411 | 97.03%           | 97.03%           |
| 70%        | 0.5115 | 90.38%           | 90.38%           |
| 90%        | 2.0637 | 62.15%           | 62.15%           |

---

## 4. Training Details

| Config             | Best Val Loss | Best Epoch | Total Epochs    | Batch Size |
| ------------------ | ------------- | ---------- | --------------- | ---------- |
| Baseline           | 3.9112        | 88         | 100             | 256        |
| +MVM Pretrain      | 3.8535        | 96         | 100             | 256        |
| +Bbox Crop         | 3.7638        | 77         | 92 (early stop) | 256        |
| +Crop+Pretrain     | 3.6589        | 79         | 94 (early stop) | 256        |
| +Crop+Pretrain+Aug | **3.6521**    | 88         | 100             | 256        |
| +Crop+Pretrain+Aug+Semantic Init     | —             | —          | —               | 256        |
| MVM Pretraining    | 0.6423        | 191        | 200             | 256        |

### Dataset

| Split | Samples |
| ----- | ------- |
| Total | 8,328   |
| Train | 6,662   |
| Val   | 833     |
| Test  | 833     |

- Voxel grid size: 32 × 32 × 32
- Block vocabulary: 256 types (mapped from 1,445+ raw IDs)
- Categories in test set: 15 unique

### Architecture

| Component        | Details                                                       |
| ---------------- | ------------------------------------------------------------- |
| Voxel Encoder    | 3D CNN: Embedding(256, 32) → Conv3d [64, 128, 256] → MLP(256) |
| Text Encoder     | Frozen all-MiniLM-L6-v2 (384-d) → MLP(256)                    |
| Shared dim       | 256                                                           |
| Loss             | CLIP-style contrastive (learnable temperature)                |
| Trainable params | 1,400,896                                                     |
| Frozen params    | 22,713,216                                                    |
