# Multimodal Minecraft Schematic Retrieval

![alt text](concept_with_builds_1779750277264.png)

## 1. Introduction

Minecraft is home to one of the largest user-generated 3D content ecosystems in the world. Community platforms like [Planet Minecraft](https://www.planetminecraft.com/) host hundreds of thousands of player-created structures, castles, houses, pixel art, redstone contraptions, entire cities, shared as downloadable **schematics** (3D voxel grids of block IDs). But finding the right schematic is painful: search is keyword-based, tags are inconsistent, and there's no way to search by _structure_ or _vibe_.

This project explores **cross-modal retrieval between natural language and 3D voxel structures** in the Minecraft domain. The idea: learn a shared embedding space where text descriptions ("medieval castle with towers") and 3D block grids live side-by-side, enabling semantic search in both directions.

The approach draws from the CLIP paradigm, a dual-encoder architecture trained with symmetric InfoNCE loss, which is then adapted for a unique setting: discrete voxel grids (not point clouds or meshes), noisy community-authored text (not curated captions), and a relatively small dataset (~8K samples). A two-stage training pipeline first pretrains the voxel encoder via masked voxel modeling (self-supervised), then fine-tunes the full dual-encoder with contrastive learning.

> [!NOTE]
> This document serves as a comprehensive explainer of the project. Covering motivation, prior work, data, use cases, architecture, and training details. The technical sections include generated diagrams and mermaid flowcharts for visual reference.

---

## 2. Prior Works

This section covers relevant prior work in cross-modal retrieval between text and 3D structures, 3D-language alignment, and related domains.

---

### 2a. TriCoLo — Trimodal Contrastive Loss for Text to Shape Retrieval

**Ruan et al., WACV 2024** — [arXiv:2201.07366](https://arxiv.org/abs/2201.07366)

TriCoLo tackles **text-to-3D shape retrieval** using a trimodal contrastive learning approach across text, multi-view images, and 3D voxels. The core insight: using rendered images as a **bridge modality** between text and 3D significantly improves retrieval, because image-text alignment is a more natural and better-studied problem than direct text-3D alignment.

**Architecture:**

- **Text encoder:** Bidirectional GRU
- **3D encoder:** 3D CNN on colored voxels
- **Image encoder:** MVCNN with pretrained ResNet18

**Loss:** A sum of three pairwise InfoNCE losses aligning all modality pairs:

```
L_tri = L(voxel, image) + L(voxel, text) + L(image, text)
```

**Key results (Text2Shape benchmark):**

| Model                        | RR@1       | RR@5       | NDCG@5     |
| ---------------------------- | ---------- | ---------- | ---------- |
| **Tri(I+V) — full trimodal** | **12.11%** | **32.39%** | **22.42%** |
| Bi(V) — text+voxel only      | 8.98%      | 26.76%     | 17.99%     |

**Relevance:** Directly comparable to our setup. TriCoLo also uses voxel-based 3D representations with contrastive alignment, and shows that simple contrastive objectives beat more complex architectures. The trimodal bridge idea is a potential future extension for our system (using rendered images of schematics as an additional modality).

---

### 2b. RI-Mamba — Rotation-Invariant Text-to-Shape Retrieval

**Nguyen et al., 2026** — [arXiv:2602.11673](https://arxiv.org/abs/2602.11673)

RI-Mamba is the **first rotation-invariant architecture based on State-Space Models (Mamba)** for text-to-3D retrieval. It addresses the realistic scenario where 3D objects appear in arbitrary orientations — existing methods assume canonical poses and break down under rotation.

**Key technical components:**

- **Local Reference Frames (LRFs):** Disentangle pose from intrinsic geometry at the input level
- **Hilbert curve serialization:** Converts unordered point cloud patches into a spatially-coherent 1D sequence for Mamba processing
- **FiLM-based orientation reintegration:** Recovers spatial context lost during LRF normalization
- **CLIP-aligned contrastive learning** with automated triplet generation (no manual annotation)

**Key results (OmniObject3D, 214 categories):**

| Method         | mAP (canonical) | mAP (SO(3) rotated) |
| -------------- | --------------- | ------------------- |
| PointBERT-TAMM | 49.84           | ~22                 |
| **RI-Mamba**   | **47.58**       | **47.50**           |

Non-rotation-invariant methods collapse from ~50→22 under rotation. RI-Mamba holds steady at ~47.5.

**Relevance:** Demonstrates state-of-the-art text→3D retrieval with a focus on robustness. The automated triplet generation strategy removes annotation bottlenecks — relevant for scaling our approach. Rotation invariance is less critical for Minecraft schematics (structures are typically axis-aligned), but the Mamba-based 3D encoding is an interesting alternative to our CNN approach.

---

### 2c. Invert3D — Aligning 3D Representations with Text Embeddings

**Song et al., ACM MM 2025** — [arXiv:2508.16932](https://arxiv.org/abs/2508.16932)

Invert3D proposes a **camera-conditioned inversion mechanism** that maps 3D scenes (NeRF / 3D Gaussian Splatting) into CLIP's text-aligned embedding space, enabling language-driven 3D editing without retraining.

**Core mechanism:**

- Renders multiple 2D views of a 3D scene from different camera poses
- Processes views through a camera-conditioned inversion module
- Produces a 3D embedding aligned with CLIP's vision-text space
- Enables latent-space manipulation for text-guided personalization

**Relevance:** While focused on editing/personalization rather than retrieval, the core technique of projecting 3D representations into a pre-aligned text embedding space is conceptually similar to our approach. The key insight — that you can leverage existing vision-language alignment (CLIP) as a bridge — is shared across many works in this space. However, Invert3D operates on continuous neural 3D representations, whereas our system uses discrete voxel grids.

---

### 2d. DreamCraft — Text-Guided 3D Generation in Minecraft

**Earle et al., FDG 2024** — [arXiv:2404.15538](https://arxiv.org/abs/2404.15538)

DreamCraft tackles **text-to-3D generation** (not retrieval) specifically in the Minecraft domain — making it the most domain-relevant prior work despite the different task formulation.

**Approach:**

- Uses a **quantized Neural Radiance Field** that learns to place discrete Minecraft block types during training (not post-hoc)
- Optimized via **Score Distillation Sampling (SDS)** against a frozen text-to-image diffusion model
- Adds **functional constraint losses**: block-type distribution targets and adjacency rules

**Relevance:** DreamCraft is complementary to our retrieval approach. Where we find _existing_ builds matching a description, DreamCraft _generates new_ builds from scratch. Interesting future work could combine both: retrieve the closest existing schematic, then use generation to modify it toward the query. The paper also validates that the Minecraft domain has real demand for text-guided 3D interaction.

> [!TIP]
> DreamCraft uses no contrastive learning or embedding alignment — the text-3D connection is purely through SDS optimization against a diffusion model. This is a fundamentally different paradigm from our retrieval-based approach.

---

### 2e. VXP — Voxel-Cross-Pixel Place Recognition

**Li et al., 3DV 2025** — [arXiv:2403.14594](https://arxiv.org/abs/2403.14594)

VXP addresses **cross-modal place recognition** between 2D camera images and 3D LiDAR point clouds (voxelized). While the application domain (autonomous driving / localization) differs from ours, the technical approach to bridging a large modality gap is highly instructive.

**Multi-stage training strategy:**

1. **Global image pre-training** — learn distinctive 2D descriptors
2. **Local correspondence alignment** — enforce feature similarity between spatially corresponding voxels and pixels via geometric projection
3. **Global descriptor consistency** — align cross-modal global embeddings in a shared space

**Key results (Oxford RobotCar):**

| Direction | Recall@1 | Recall@1% |
| --------- | -------- | --------- |
| 2D → 3D   | 47.16%   | 71.72%    |
| 3D → 2D   | 30.01%   | 56.09%    |

**Relevance:** VXP's key takeaway for our project is that bridging a large modality gap benefits from **local-to-global alignment** rather than just contrasting global descriptors. Their multi-stage training (local features first, then global) parallels our two-stage approach (MVM pretraining for local voxel understanding first, then global contrastive alignment). The voxel-based 3D processing is also architecturally similar to our VoxelEncoder.

---

### Summary of Prior Work Landscape

```mermaid
graph TD
    subgraph "Text ↔ 3D Retrieval (Direct)"
        TR1["TriCoLo<br/>(Voxel + Image + Text)<br/>WACV 2024"]
        TR2["RI-Mamba<br/>(Point Cloud + Text)<br/>2026"]
    end

    subgraph "3D-Language Alignment"
        AL1["Invert3D<br/>(NeRF/3DGS → CLIP space)<br/>MM 2025"]
    end

    subgraph "Text → 3D Generation"
        GEN1["DreamCraft<br/>(Minecraft NeRF + SDS)<br/>FDG 2024"]
    end

    subgraph "Cross-Modal 3D Retrieval (Other)"
        CM1["VXP<br/>(Image ↔ LiDAR Voxels)<br/>3DV 2025"]
    end

    subgraph "Ours"
        OURS["MC-Retrieval<br/>(Text ↔ Minecraft Voxels)<br/>CLIP-style + MVM"]
    end

    TR1 -.- |"shared: contrastive<br/>voxel encoding"| OURS
    TR2 -.- |"shared: text→3D<br/>retrieval task"| OURS
    AL1 -.- |"shared: 3D→text<br/>embedding alignment"| OURS
    GEN1 -.- |"shared: Minecraft<br/>domain"| OURS
    CM1 -.- |"shared: multi-stage<br/>voxel alignment"| OURS

    style OURS fill:#7c3aed,stroke:#5b21b6,color:#fff
    style TR1 fill:#2563eb,stroke:#1d4ed8,color:#fff
    style TR2 fill:#2563eb,stroke:#1d4ed8,color:#fff
    style AL1 fill:#059669,stroke:#047857,color:#fff
    style GEN1 fill:#ea580c,stroke:#c2410c,color:#fff
    style CM1 fill:#0891b2,stroke:#0e7490,color:#fff
```

> [!IMPORTANT]
> Our work occupies a unique niche: **text ↔ discrete voxel retrieval in a creative/gaming domain**. Most prior text-3D retrieval work targets clean CAD models (ShapeNet) or real-world scans. Our dataset of noisy, diverse, community-created Minecraft schematics presents distinct challenges — highly variable quality, creative naming conventions, and a discrete block vocabulary — that aren't addressed by existing methods.

---

## 3. Data Source

The dataset powering this system is a collection of **8,328 Minecraft schematics** scraped from [Planet Minecraft](https://www.planetminecraft.com/), one of the largest community hubs for user-created Minecraft content. Each record pairs a 3D voxel structure with rich user-authored metadata.

### What's in the dataset

| Feature                  | Type    | Description                                                            |
| ------------------------ | ------- | ---------------------------------------------------------------------- |
| `voxel_data`             | object  | Flattened 32×32×32 array of Minecraft block IDs — the raw 3D structure |
| `title`                  | object  | User-given name (e.g. "Medieval Castle", "Cozy Treehouse")             |
| `subtitle`               | object  | Short tagline or secondary title                                       |
| `description`            | object  | Free-form HTML description written by the creator                      |
| `tags`                   | object  | Community/creator-assigned category tags (JSON-serialized list)        |
| `img`                    | object  | Thumbnail image URL                                                    |
| `bigImgs`                | object  | Gallery image URLs (JSON-serialized)                                   |
| `user`                   | object  | Creator username                                                       |
| `date`                   | object  | Upload date                                                            |
| `diamondCount`           | int64   | Community "diamond" upvotes                                            |
| `views`                  | int64   | View count                                                             |
| `downloads`              | int64   | Download count                                                         |
| `comments`               | float64 | Number of comments                                                     |
| `favorites`              | int64   | Favorite count                                                         |
| `url`                    | object  | Original Planet Minecraft page URL                                     |
| `downloadLink`           | object  | Primary download link                                                  |
| `finalDownloadLink`      | object  | Resolved download link                                                 |
| `thirdPartyDownloadLink` | object  | External mirror link                                                   |
| `youtubeId`              | object  | Associated YouTube showcase video ID                                   |

- **Format:** Apache Parquet (`data.parquet`, ~11 MB)
- **Origin:** Originally collected by Romain Beaumont's [minecraft-schematics-dataset](https://github.com/rom1504/minecraft-schematics-dataset) project, reformatted into parquet for ease of use.

### How the text modality is constructed

The text input for each sample is assembled by concatenating four metadata fields:

```
title + subtitle + description (HTML-stripped) + tags
```

This produces a natural-language-ish description that captures the creator's intent, structural details, and categorical context, all without requiring any manual annotation.

### Voxel representation

Each schematic is stored as a flat array of block IDs representing a 32³ voxel grid. During preprocessing, the top-254 most frequent block types are kept (mapped to IDs 2–255), with `0 = air` and `1 = rare/other`. The grid is bounding-box cropped to the non-air region, then nearest-neighbor resized back to 32³.

> [!NOTE]
> The dataset is entirely **community-generated**. Structures range from tiny furniture pieces to sprawling castles, with highly variable quality, style, and complexity. This makes it a challenging but realistic benchmark for cross-modal retrieval.

---

## 4. Use Cases

### 4a. Text-to-Build Search

> _"I want a medieval castle with towers and a moat"_

The most straightforward application: a player types a natural language description and the system retrieves the most similar schematics from the database, ranked by cosine similarity. This replaces the current keyword-based search on platforms like Planet Minecraft, which struggles with semantic queries (e.g. "cozy" or "futuristic" don't map to specific block types).

### 4b. Build-to-Text Discovery (Reverse Search)

Given a voxel structure (e.g. one you just built), retrieve the closest text descriptions, essentially asking _"what does this look like?"_ or _"what would someone call this?"_. Useful for:

- Auto-generating titles/tags for uploads
- Finding similar existing builds to compare against
- Content moderation (detecting copies or near-duplicates)

### 4c. Recommendation & Similarity Browsing

Since both modalities live in the same embedding space, you can use voxel→voxel similarity to power a **"more like this"** recommendation system. A user clicks on a build they like, and the system finds structurally similar ones without relying on tags or metadata at all.

### 4d. Content Organization & Clustering

The learned embeddings can be used to **automatically cluster** the schematic database into semantic groups (castles, houses, vehicles, pixel art, redstone machines, etc.) without manual labeling. This enables:

- Better browse/filter UIs for schematic repositories
- Automatic category assignment for new uploads
- Dataset curation and quality filtering

### 4e. Creative Assistance & Inspiration

Builders looking for inspiration can describe a vague concept and get back real examples that match the vibe. Unlike generation-based approaches (which create _new_ structures), retrieval surfaces _existing_ community creations, often with download links, tutorials, and creator commentary attached.

### 4f. Integration with Minecraft Modding Tools

The retrieval system could be embedded into:

- **WorldEdit / Litematica plugins** — search and paste schematics from within the game
- **Web-based schematic browsers** — semantic search API for community platforms
- **Educational tools** — find example builds for teaching architectural or engineering concepts in Minecraft

---

## 5. High-Level Concept

![Cross-modal retrieval between text and 3D voxel structures in a shared embedding space](./concept_overview_v2_1779725761167.png)

The core idea: learn a **shared embedding space** where text descriptions and 3D voxel structures live side-by-side, enabling cross-modal search in both directions.

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

## 6. Full System Flowchart

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

## 7. Model Architecture

![DualEncoder architecture with VoxelEncoder (3D CNN) and TextEncoder (frozen SentenceTransformer + projection)](./architecture_v2_1779725772833.png)

### 7a. DualEncoder — Top-Level

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

### 7b. VoxelEncoder — 3D CNN Pipeline

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

### 7c. TextEncoder — Frozen Backbone + Learned Projection

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

## 8. Training Mechanism

### 8a. Stage 1 — Masked Voxel Modeling (Self-Supervised Pretraining)

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

### 8b. Stage 2 — CLIP-Style Contrastive Fine-Tuning

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

### 8c. Two-Stage Training Overview

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

## 9. Data Preprocessing Pipeline

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

## 10. Key Hyperparameters Summary

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
