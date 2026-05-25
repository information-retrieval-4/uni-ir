# Multimodal Minecraft Schematic Retrieval — Diagrams & Figures

---

## 1. High-Level Concept

![Cross-modal retrieval between text and 3D voxel structures in a shared embedding space](./concept_overview_v2_1779725761167.png)

The core idea: learn a **shared embedding space** where text descriptions and 3D voxel structures live side-by-side — enabling cross-modal search in both directions.

```mermaid
graph LR
    subgraph Query Space
        Q1["🔍 'medieval castle<br/>with towers'"]
        Q2["🔍 'small wooden<br/>house'"]
        Q3["🧊 Voxel Grid<br/>32³ blocks"]
    end

    subgraph Shared Embedding Space
        direction TB
        E["256-dim<br/>L2-normalized<br/>vector space"]
    end

    subgraph Retrieved Results
        R1["🧊 Castle Schematic"]
        R2["🧊 House Schematic"]
        R3["📝 'Cozy cabin<br/>with chimney'"]
    end

    Q1 -->|Text Encoder| E
    Q2 -->|Text Encoder| E
    Q3 -->|Voxel Encoder| E
    E -->|cosine sim| R1
    E -->|cosine sim| R2
    E -->|cosine sim| R3

    style E fill:#7c3aed,stroke:#5b21b6,color:#fff
    style Q1 fill:#1e3a5f,stroke:#2563eb,color:#fff
    style Q2 fill:#1e3a5f,stroke:#2563eb,color:#fff
    style Q3 fill:#064e3b,stroke:#059669,color:#fff
    style R1 fill:#064e3b,stroke:#059669,color:#fff
    style R2 fill:#064e3b,stroke:#059669,color:#fff
    style R3 fill:#1e3a5f,stroke:#2563eb,color:#fff
```

> [!NOTE]
> The system supports **bidirectional retrieval**: Text→Voxel (find structures from descriptions) and Voxel→Text (find descriptions for structures).

---

## 2. Full System Flowchart

End-to-end pipeline from raw data to retrieval results.

```mermaid
flowchart TD
    subgraph DATA["📦 Data Layer"]
        PMC["Planet Minecraft<br/>Scrape"]
        PQ["data.parquet<br/>8,328 samples"]
        PMC --> PQ
    end

    subgraph PREPROC["⚙️ Preprocessing"]
        BM["Block Mapping<br/>top-256 IDs<br/>0=air, 1=rare"]
        BBOX["Bounding Box Crop<br/>+ NN Resize → 32³"]
        TXT["Text Assembly<br/>title + subtitle +<br/>description + tags"]
        PQ --> BM
        PQ --> BBOX
        PQ --> TXT
        BM --> BBOX
    end

    subgraph STAGE1["🔧 Stage 1: Self-Supervised Pretraining"]
        MVM["Masked Voxel<br/>Modeling"]
        UNET["U-Net Encoder-Decoder<br/>mask 20% non-air blocks<br/>→ reconstruct"]
        MVM --> UNET
        BBOX --> MVM
    end

    subgraph STAGE2["🎯 Stage 2: Contrastive Fine-Tuning"]
        DUAL["DualEncoder<br/>VoxelEncoder + TextEncoder"]
        CLIP["CLIP-style InfoNCE<br/>symmetric loss"]
        UNET -->|"transfer encoder<br/>weights"| DUAL
        BBOX --> DUAL
        TXT --> DUAL
        DUAL --> CLIP
    end

    subgraph EVAL["📊 Evaluation"]
        EMB["Extract Embeddings<br/>text_emb, voxel_emb"]
        MET["Metrics<br/>R@1, R@5, R@10<br/>MRR, Median Rank"]
        CAT["Category Metrics<br/>Precision@k<br/>Hit Rate@k"]
        CLIP --> EMB
        EMB --> MET
        EMB --> CAT
    end

    subgraph INFER["🔍 Inference"]
        SEARCH["Cosine Similarity<br/>Nearest Neighbor<br/>Retrieval"]
        EMB --> SEARCH
    end

    style DATA fill:#1e293b,stroke:#475569,color:#e2e8f0
    style PREPROC fill:#1e293b,stroke:#475569,color:#e2e8f0
    style STAGE1 fill:#1a1a2e,stroke:#6366f1,color:#e2e8f0
    style STAGE2 fill:#1a1a2e,stroke:#8b5cf6,color:#e2e8f0
    style EVAL fill:#1a1a2e,stroke:#10b981,color:#e2e8f0
    style INFER fill:#1a1a2e,stroke:#f59e0b,color:#e2e8f0
```

---

## 3. Model Architecture

![DualEncoder architecture with VoxelEncoder (3D CNN) and TextEncoder (frozen SentenceTransformer + projection)](./architecture_v2_1779725772833.png)

### 3a. DualEncoder — Top-Level

```mermaid
flowchart LR
    subgraph Inputs
        T["📝 Text<br/>(list of strings)"]
        V["🧊 Voxels<br/>(B, 32, 32, 32)<br/>LongTensor"]
    end

    subgraph DualEncoder
        direction TB
        TE["TextEncoder"]
        VE["VoxelEncoder"]

        TE --> NormT["L2 Normalize"]
        VE --> NormV["L2 Normalize"]
    end

    T --> TE
    V --> VE

    NormT --> TE_OUT["text_emb<br/>(B, 256)"]
    NormV --> VE_OUT["voxel_emb<br/>(B, 256)"]

    style TE fill:#2563eb,stroke:#1d4ed8,color:#fff
    style VE fill:#059669,stroke:#047857,color:#fff
    style NormT fill:#4f46e5,stroke:#4338ca,color:#fff
    style NormV fill:#4f46e5,stroke:#4338ca,color:#fff
    style TE_OUT fill:#1e3a5f,stroke:#2563eb,color:#fff
    style VE_OUT fill:#064e3b,stroke:#059669,color:#fff
```

---

### 3b. VoxelEncoder — 3D CNN Pipeline

```mermaid
flowchart TD
    IN["Input<br/>(B, 32, 32, 32)<br/>Block IDs"]

    EMB["nn.Embedding<br/>256 block types → 64-dim<br/>(B, 32, 32, 32, 64)"]

    PERM["Permute<br/>(B, 64, 32, 32, 32)"]

    subgraph ConvStack["3D CNN Stack"]
        direction TB
        B1["Conv3d 64→128, k=3<br/>BatchNorm3d + GELU<br/>Dropout3d + MaxPool3d(2)<br/>→ (B, 128, 16, 16, 16)"]

        B2["Conv3d 128→256, k=3<br/>BatchNorm3d + GELU<br/>Dropout3d + MaxPool3d(2)<br/>→ (B, 256, 8, 8, 8)"]

        B3["Conv3d 256→512, k=3<br/>BatchNorm3d + GELU<br/>Dropout3d + AdaptiveAvgPool3d(1)<br/>→ (B, 512, 1, 1, 1)"]

        B1 --> B2 --> B3
    end

    FLAT["Flatten<br/>(B, 512)"]

    PROJ["Linear 512→256<br/>+ Dropout"]

    OUT["Output<br/>(B, 256)"]

    IN --> EMB --> PERM --> B1
    B3 --> FLAT --> PROJ --> OUT

    style IN fill:#064e3b,stroke:#059669,color:#fff
    style EMB fill:#155e75,stroke:#0891b2,color:#fff
    style PERM fill:#334155,stroke:#64748b,color:#e2e8f0
    style ConvStack fill:#1e1e2e,stroke:#6366f1,color:#e2e8f0
    style B1 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style B2 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style B3 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style FLAT fill:#334155,stroke:#64748b,color:#e2e8f0
    style PROJ fill:#7c2d12,stroke:#ea580c,color:#fff
    style OUT fill:#064e3b,stroke:#059669,color:#fff
```

---

### 3c. TextEncoder — Frozen Backbone + Learned Projection

```mermaid
flowchart TD
    IN["Input<br/>list of B strings"]

    subgraph Frozen["❄️ Frozen Backbone"]
        ST["SentenceTransformer<br/>all-MiniLM-L6-v2<br/>(~22M params, frozen)"]
    end

    FEAT["Text Features<br/>(B, 384)"]

    subgraph Proj["🔥 Learned Projection Head"]
        L1["Linear 384→256"]
        G1["GELU"]
        D1["Dropout 0.3"]
        L2["Linear 256→256"]
        L1 --> G1 --> D1 --> L2
    end

    OUT["Output<br/>(B, 256)"]

    IN --> ST --> FEAT --> L1
    L2 --> OUT

    style IN fill:#1e3a5f,stroke:#2563eb,color:#fff
    style Frozen fill:#1e293b,stroke:#94a3b8,color:#94a3b8
    style ST fill:#374151,stroke:#94a3b8,color:#e2e8f0
    style FEAT fill:#334155,stroke:#64748b,color:#e2e8f0
    style Proj fill:#1e1e2e,stroke:#f59e0b,color:#e2e8f0
    style L1 fill:#7c2d12,stroke:#ea580c,color:#fff
    style G1 fill:#4a1d96,stroke:#8b5cf6,color:#fff
    style D1 fill:#334155,stroke:#64748b,color:#e2e8f0
    style L2 fill:#7c2d12,stroke:#ea580c,color:#fff
    style OUT fill:#1e3a5f,stroke:#2563eb,color:#fff
```

---

## 4. Training Mechanism

### 4a. Stage 1 — Masked Voxel Modeling (Self-Supervised Pretraining)

![MVM pretraining: mask 20% non-air blocks, U-Net encoder-decoder reconstructs them](./mvm_pretraining_v2_1779725788186.png)

```mermaid
flowchart TD
    IN["Input Voxel Grid<br/>(B, 32, 32, 32)"]

    AUG["Augmentation<br/>random 90° Y-rotation<br/>+ horizontal flips"]

    MASK["Masking<br/>20% of non-air blocks<br/>replaced with [MASK] token"]

    subgraph UNet["U-Net Encoder-Decoder"]
        direction TB

        subgraph Encoder
            direction TB
            EMB2["Block Embedding<br/>(+1 for mask token)<br/>→ (B, D, 32³)"]
            E1["Enc Block 1<br/>Conv3d → BN → GELU → Drop<br/>→ (B, 64, 32³)"]
            P1["MaxPool3d(2)<br/>→ 16³"]
            E2["Enc Block 2<br/>Conv3d → BN → GELU → Drop<br/>→ (B, 128, 16³)"]
            P2["MaxPool3d(2)<br/>→ 8³"]
            BN["Bottleneck<br/>Conv3d → BN → GELU → Drop<br/>→ (B, 256, 8³)"]
            EMB2 --> E1 --> P1 --> E2 --> P2 --> BN
        end

        subgraph Decoder
            direction TB
            U2["ConvTranspose3d<br/>→ (B, 128, 16³)"]
            CAT2["Concat + Conv<br/>skip from E2<br/>→ (B, 128, 16³)"]
            U1["ConvTranspose3d<br/>→ (B, 64, 32³)"]
            CAT1["Concat + Conv<br/>skip from E1<br/>→ (B, 64, 32³)"]
            HEAD["Conv3d 1×1<br/>→ (B, num_blocks, 32³)"]
            U2 --> CAT2 --> U1 --> CAT1 --> HEAD
        end

        BN --> U2
        E2 -.->|skip| CAT2
        E1 -.->|skip| CAT1
    end

    LOSS1["Cross-Entropy Loss<br/>(only on masked positions)"]

    IN --> AUG --> MASK --> EMB2
    HEAD --> LOSS1

    style IN fill:#064e3b,stroke:#059669,color:#fff
    style AUG fill:#4a1d96,stroke:#8b5cf6,color:#fff
    style MASK fill:#7f1d1d,stroke:#dc2626,color:#fff
    style UNet fill:#0f172a,stroke:#6366f1,color:#e2e8f0
    style Encoder fill:#1e1e2e,stroke:#3b82f6,color:#e2e8f0
    style Decoder fill:#1e1e2e,stroke:#f97316,color:#e2e8f0
    style EMB2 fill:#155e75,stroke:#0891b2,color:#fff
    style E1 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style P1 fill:#334155,stroke:#64748b,color:#e2e8f0
    style E2 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style P2 fill:#334155,stroke:#64748b,color:#e2e8f0
    style BN fill:#312e81,stroke:#818cf8,color:#e2e8f0
    style U2 fill:#7c2d12,stroke:#ea580c,color:#fff
    style CAT2 fill:#7c2d12,stroke:#ea580c,color:#fff
    style U1 fill:#7c2d12,stroke:#ea580c,color:#fff
    style CAT1 fill:#7c2d12,stroke:#ea580c,color:#fff
    style HEAD fill:#7c2d12,stroke:#ea580c,color:#fff
    style LOSS1 fill:#7f1d1d,stroke:#dc2626,color:#fff
```

> [!IMPORTANT]
> After MVM pretraining, the **encoder weights are extracted** using `get_encoder_state_dict()` and transferred to the VoxelEncoder in the DualEncoder. The decoder is discarded.

---

### 4b. Stage 2 — CLIP-Style Contrastive Fine-Tuning

![CLIP-style symmetric InfoNCE with similarity matrix and learnable temperature](./contrastive_training_v2_1779725800763.png)

```mermaid
flowchart TD
    subgraph Batch["Mini-Batch (B samples)"]
        T["📝 B texts"]
        V["🧊 B voxel grids"]
    end

    subgraph Model["DualEncoder"]
        TE["TextEncoder<br/>(frozen backbone<br/>+ trainable proj)"]
        VE["VoxelEncoder<br/>(pretrained from MVM<br/>+ trainable proj)"]
    end

    T --> TE
    V --> VE

    TE --> T_EMB["text_emb<br/>(B, 256)<br/>L2-normed"]
    VE --> V_EMB["voxel_emb<br/>(B, 256)<br/>L2-normed"]

    subgraph Loss["Symmetric InfoNCE Loss"]
        SIM["Similarity Matrix<br/>S = T @ V^T / τ<br/>(B × B)"]
        T2V["CE(S, labels)<br/>Text → Voxel"]
        V2T["CE(S^T, labels)<br/>Voxel → Text"]
        AVG["Loss = (L_t2v + L_v2t) / 2"]
        SIM --> T2V
        SIM --> V2T
        T2V --> AVG
        V2T --> AVG
    end

    T_EMB --> SIM
    V_EMB --> SIM

    AVG --> BP["Backprop<br/>+ grad clipping"]

    style Batch fill:#1e293b,stroke:#475569,color:#e2e8f0
    style T fill:#1e3a5f,stroke:#2563eb,color:#fff
    style V fill:#064e3b,stroke:#059669,color:#fff
    style Model fill:#0f172a,stroke:#8b5cf6,color:#e2e8f0
    style TE fill:#2563eb,stroke:#1d4ed8,color:#fff
    style VE fill:#059669,stroke:#047857,color:#fff
    style T_EMB fill:#1e3a5f,stroke:#2563eb,color:#fff
    style V_EMB fill:#064e3b,stroke:#059669,color:#fff
    style Loss fill:#1e1e2e,stroke:#f59e0b,color:#e2e8f0
    style SIM fill:#7c2d12,stroke:#ea580c,color:#fff
    style T2V fill:#7f1d1d,stroke:#dc2626,color:#fff
    style V2T fill:#7f1d1d,stroke:#dc2626,color:#fff
    style AVG fill:#7f1d1d,stroke:#ef4444,color:#fff
    style BP fill:#4a1d96,stroke:#8b5cf6,color:#fff
```

#### The Similarity Matrix

```
              Voxel₁  Voxel₂  Voxel₃  ...  VoxelB
    Text₁  │  ✓ 0.92   0.11    0.05         0.03  │
    Text₂  │    0.08  ✓ 0.89   0.12         0.07  │
    Text₃  │    0.04    0.06  ✓ 0.87         0.10  │
     ...   │    ...     ...     ...          ...   │
    TextB  │    0.02    0.05    0.03       ✓ 0.91  │

    ✓ = diagonal = ground-truth match (labels = [0, 1, 2, ..., B-1])
    τ = learnable temperature, clamped to [0.01, 1.0]
```

---

### 4c. Two-Stage Training Overview

![Two-stage pipeline: MVM pretraining → weight transfer → contrastive fine-tuning → evaluation](./training_pipeline_v2_1779725813631.png)

```mermaid
flowchart LR
    subgraph S1["Stage 1: MVM Pretraining"]
        direction TB
        D1["All 8,328 samples<br/>(voxels only)"]
        M1["U-Net<br/>Encoder-Decoder"]
        O1["200 epochs<br/>lr=1e-3<br/>AdamW + CosineAnnealing"]
        D1 --> M1 --> O1
    end

    TRANSFER["🔀 Weight Transfer<br/>Encoder weights →<br/>VoxelEncoder"]

    subgraph S2["Stage 2: Contrastive Training"]
        direction TB
        D2["Train/Val/Test split<br/>80/10/10<br/>(text + voxels)"]
        M2["DualEncoder<br/>+ CLIPLoss"]
        O2["100 epochs<br/>lr_voxel=3e-4, lr_proj=1e-4<br/>early stopping (15)"]
        D2 --> M2 --> O2
    end

    EVAL["📊 Evaluation<br/>R@1, R@5, R@10<br/>MRR, MedR<br/>Category metrics"]

    S1 --> TRANSFER --> S2 --> EVAL

    style S1 fill:#1a1a2e,stroke:#6366f1,color:#e2e8f0
    style S2 fill:#1a1a2e,stroke:#8b5cf6,color:#e2e8f0
    style TRANSFER fill:#7c2d12,stroke:#f59e0b,color:#fff
    style EVAL fill:#064e3b,stroke:#10b981,color:#fff
    style D1 fill:#334155,stroke:#64748b,color:#e2e8f0
    style M1 fill:#312e81,stroke:#6366f1,color:#e2e8f0
    style O1 fill:#334155,stroke:#64748b,color:#e2e8f0
    style D2 fill:#334155,stroke:#64748b,color:#e2e8f0
    style M2 fill:#312e81,stroke:#8b5cf6,color:#e2e8f0
    style O2 fill:#334155,stroke:#64748b,color:#e2e8f0
```

---

## 5. Data Preprocessing Pipeline

```mermaid
flowchart TD
    RAW["Raw Parquet<br/>8,328 records<br/>19 features"]

    subgraph VoxelPreproc["Voxel Preprocessing"]
        direction TB
        VP1["Flat array → 32³ grid"]
        VP2["Build block mapping<br/>top-254 frequent blocks<br/>0=air, 1=rare"]
        VP3["Remap block IDs"]
        VP4["Bounding-box crop<br/>(non-air region)"]
        VP5["Nearest-neighbor resize<br/>back to 32³"]
        VP1 --> VP2 --> VP3 --> VP4 --> VP5
    end

    subgraph TextPreproc["Text Preprocessing"]
        direction TB
        TP1["Strip HTML tags"]
        TP2["Collapse whitespace"]
        TP3["Concatenate fields:<br/>title + subtitle +<br/>description + tags"]
        TP1 --> TP2 --> TP3
    end

    subgraph Splitting["Data Splits"]
        direction LR
        TR["Train 80%<br/>~6,662"]
        VA["Val 10%<br/>~833"]
        TE["Test 10%<br/>~833"]
    end

    RAW --> VoxelPreproc
    RAW --> TextPreproc
    VoxelPreproc --> Splitting
    TextPreproc --> Splitting

    style RAW fill:#1e293b,stroke:#475569,color:#e2e8f0
    style VoxelPreproc fill:#0f172a,stroke:#059669,color:#e2e8f0
    style TextPreproc fill:#0f172a,stroke:#2563eb,color:#e2e8f0
    style Splitting fill:#0f172a,stroke:#f59e0b,color:#e2e8f0
    style VP1 fill:#064e3b,stroke:#059669,color:#fff
    style VP2 fill:#064e3b,stroke:#059669,color:#fff
    style VP3 fill:#064e3b,stroke:#059669,color:#fff
    style VP4 fill:#064e3b,stroke:#059669,color:#fff
    style VP5 fill:#064e3b,stroke:#059669,color:#fff
    style TP1 fill:#1e3a5f,stroke:#2563eb,color:#fff
    style TP2 fill:#1e3a5f,stroke:#2563eb,color:#fff
    style TP3 fill:#1e3a5f,stroke:#2563eb,color:#fff
    style TR fill:#7c2d12,stroke:#ea580c,color:#fff
    style VA fill:#7c2d12,stroke:#f59e0b,color:#fff
    style TE fill:#7c2d12,stroke:#f59e0b,color:#fff
```

---

## 6. Key Hyperparameters Summary

| Component                | Parameter        | Value            |
| ------------------------ | ---------------- | ---------------- |
| **Embedding Space**      | Dimension        | 256              |
| **VoxelEncoder**         | Block embed dim  | 32               |
|                          | CNN channels     | [64, 128, 256]   |
|                          | Dropout          | 0.3              |
| **TextEncoder**          | Backbone         | all-MiniLM-L6-v2 |
|                          | Hidden dim       | 384              |
|                          | Backbone frozen  | ✓                |
| **MVM Pretraining**      | Mask ratio       | 20%              |
|                          | Epochs           | 200              |
|                          | Learning rate    | 1e-3             |
|                          | Batch size       | 256              |
| **Contrastive Training** | Epochs           | 100              |
|                          | LR (voxel)       | 3e-4             |
|                          | LR (text proj)   | 1e-4             |
|                          | Temperature init | 0.07 (learnable) |
|                          | Early stopping   | 15 epochs        |
|                          | Batch size       | 256              |
| **Data**                 | Samples          | 8,328            |
|                          | Block vocab      | 256              |
|                          | Voxel grid       | 32³              |
|                          | Splits           | 80/10/10         |
