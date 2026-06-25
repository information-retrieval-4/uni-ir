"""Model architectures: VoxelEncoder, TextEncoder, Point-BERT Encoder, DualEncoder."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from timm.layers import DropPath, trunc_normal_
import open_clip


def apply_semantic_init(
    voxel_embedding_layer: nn.Embedding,
    text_encoder,
    block_names: list[str],
    block_embed_dim: int,
    device: torch.device,
):
    """Initialize voxel embedding with PCA-reduced text embeddings of block names."""
    print(f"Applying semantic block initialization to {len(block_names)} blocks...")
    with torch.no_grad():
        text_feats = text_encoder.encode_text(block_names)
        text_feats = text_feats.to(device)
        U, S, V = torch.pca_lowrank(text_feats, q=block_embed_dim)
        # U * S is equivalent to (text_feats - text_feats.mean(0)) @ V
        reduced = U * S

        # Scale to match standard embedding init variance (~0.01)
        # taking global std() preserves the relative importance of PCA components
        reduced = reduced / (reduced.std() + 1e-8)
        reduced = reduced * 0.1

        voxel_embedding_layer.weight.data.copy_(reduced)


# ---------------------------------------------------------------------------
# Depthwise Separable 3D Convolution
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# Voxel Encoder — Block Embedding + 3D CNN
# ---------------------------------------------------------------------------


class VoxelEncoder(nn.Module):
    """Encodes a 32×32×32 block-ID grid into a dense embedding vector.

    Pipeline:
        (B, 32, 32, 32) LongTensor
        → nn.Embedding  →  (B, 32, 32, 32, block_embed_dim)
        → permute        →  (B, block_embed_dim, 32, 32, 32)
        → Conv3d stack   →  (B, C, 1, 1, 1)
        → project        →  (B, embed_dim)
    """

    def __init__(
        self,
        num_block_types: int = 256,
        block_embed_dim: int = 64,
        channels: list[int] = [128, 256, 512],
        embed_dim: int = 256,
        dropout: float = 0.3,
        use_learned_stem: bool = False,
        use_depthwise_separable: bool = False,
    ):
        super().__init__()

        self.block_embedding = nn.Embedding(num_block_types, block_embed_dim)

        layers = []
        in_ch = block_embed_dim

        if use_learned_stem:
            layers.extend(
                [
                    nn.Conv3d(in_ch, in_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(in_ch),
                    nn.GELU(),
                ]
            )

        for out_ch in channels[:-1]:
            conv_layer = (
                DepthwiseSeparableConv3d(in_ch, out_ch, kernel_size=3, padding=1)
                if use_depthwise_separable
                else nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            layers.extend(
                [
                    conv_layer,
                    nn.BatchNorm3d(out_ch),
                    nn.GELU(),
                    nn.Dropout3d(dropout),
                    nn.MaxPool3d(2),
                ]
            )
            in_ch = out_ch

        # last conv block uses adaptive pooling instead of maxpool
        last_conv_layer = (
            DepthwiseSeparableConv3d(in_ch, channels[-1], kernel_size=3, padding=1)
            if use_depthwise_separable
            else nn.Conv3d(in_ch, channels[-1], kernel_size=3, padding=1)
        )
        layers.extend(
            [
                last_conv_layer,
                nn.BatchNorm3d(channels[-1]),
                nn.GELU(),
                nn.Dropout3d(dropout),
                nn.AdaptiveAvgPool3d(1),
            ]
        )

        self.conv_stack = nn.Sequential(*layers)
        self.project = nn.Sequential(
            nn.Linear(channels[-1], embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            voxels: (B, 32, 32, 32) block ID tensor
        Returns:
            (B, embed_dim) L2-normalised embedding
        """
        x = self.block_embedding(voxels)  # (B, 32, 32, 32, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, 32, 32, 32)
        x = self.conv_stack(x)  # (B, C, 1, 1, 1)
        x = x.flatten(1)  # (B, C)
        x = self.project(x)  # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Point-BERT Components
# ---------------------------------------------------------------------------


def fps_sample(xyz: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest Point Sampling → indices (n_samples,)."""
    N = len(xyz)
    if N == 0:
        return np.zeros(n_samples, dtype=np.int64)
    if N <= n_samples:
        idx = np.arange(N)
        pad = np.random.choice(N, n_samples - N, replace=True)
        return np.concatenate([idx, pad])

    selected = np.zeros(n_samples, dtype=np.int64)
    dist = np.full(N, np.inf)
    farthest = np.random.randint(N)

    for i in range(n_samples):
        selected[i] = farthest
        d = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))

    return selected


class VoxelToPoints(nn.Module):
    """Konversi voxel 32³ → point cloud sparse berukuran tetap M."""

    def __init__(self, num_points: int = 512, use_fps_eval: bool = True):
        super().__init__()
        self.num_points = num_points
        self.use_fps_eval = use_fps_eval

    def _process_one(self, grid: torch.LongTensor):
        device = grid.device
        M = self.num_points

        non_air = grid != 0
        coords = non_air.nonzero(as_tuple=False).float()  # (N, 3)
        bids = grid[non_air]  # (N,)
        N = len(coords)

        if N == 0:
            return (
                torch.zeros(M, 3, device=device),
                torch.zeros(M, dtype=torch.long, device=device),
            )

        # Zero-center + normalisasi [-1, 1]
        centroid = coords.mean(dim=0, keepdim=True)
        xyz = coords - centroid
        scale = xyz.abs().max().clamp(min=1.0)
        xyz = xyz / scale

        # Sample M titik
        if N >= M:
            use_fps = self.use_fps_eval and (not self.training)
            if use_fps:
                idx = fps_sample(xyz.cpu().numpy(), M)
                idx = torch.from_numpy(idx).long().to(device)
            else:
                idx = torch.randperm(N, device=device)[:M]
        else:
            reps = (M + N - 1) // N
            idx = torch.arange(N, device=device).repeat(reps)[:M]

        return xyz[idx], bids[idx]

    def forward(self, voxels: torch.LongTensor):
        xyz_list, bid_list = [], []
        for b in range(voxels.shape[0]):
            xyz, bids = self._process_one(voxels[b])
            xyz_list.append(xyz)
            bid_list.append(bids)
        return torch.stack(xyz_list), torch.stack(bid_list)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointBERTTransformer(nn.Module):
    def __init__(
        self,
        trans_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.trans_dim = trans_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_point_bert_checkpoint(self, ckpt_path: str):
        print(f"[PointBERT] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "base_model" in ckpt:
            raw = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        else:
            raw = {k.replace("module.", ""): v for k, v in ckpt.items()}

        base = {}
        for k, v in raw.items():
            if k.startswith("transformer_q.") and not k.startswith(
                "transformer_q.cls_head"
            ):
                base[k[len("transformer_q.") :]] = v
            elif k.startswith("base_model."):
                base[k[len("base_model.") :]] = v

        if not base:
            base = raw

        keep_prefixes = (
            "blocks.",
            "norm.",
            "cls_token",
            "cls_pos",
            "pos_embed.",
        )
        filtered = {
            k: v
            for k, v in base.items()
            if any(k.startswith(p) for p in keep_prefixes)
            or k in ("cls_token", "cls_pos")
        }

        if not filtered:
            print("[PointBERT] WARNING: Tidak ada kunci yang cocok!")
            return

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(
            f"[PointBERT] Checkpoint loaded — matched: {loaded}/{len(filtered)}  missing: {len(missing)}  unexpected: {len(unexpected)}"
        )

    def load_ulip_checkpoint(self, ckpt_path: str):
        """Load PointBERT backbone weights from a ULIP-2 checkpoint.

        ULIP stores the 3D encoder under 'point_encoder' key inside the state dict.
        We strip that prefix and load only transformer blocks/norm/pos/cls weights.
        """
        print(f"[PointBERT] Loading ULIP-2 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # ULIP checkpoints are usually {"state_dict": {...}} or flat
        raw = ckpt.get("state_dict", ckpt)
        raw = {k.replace("module.", ""): v for k, v in raw.items()}

        # Extract keys under 'point_encoder.' prefix
        prefix = "point_encoder."
        base = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}

        if not base:
            # Fallback: try without prefix (some ULIP variants store flat)
            print("[PointBERT] 'point_encoder' prefix not found, trying flat keys...")
            base = raw

        keep_prefixes = ("blocks.", "norm.", "cls_token", "cls_pos", "pos_embed.")
        filtered = {
            k: v
            for k, v in base.items()
            if any(k.startswith(p) for p in keep_prefixes)
            or k in ("cls_token", "cls_pos")
        }

        if not filtered:
            print("[PointBERT] WARNING: Tidak ada kunci ULIP yang cocok! Cek struktur checkpoint.")
            return

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(
            f"[PointBERT] ULIP checkpoint loaded — matched: {loaded}/{len(filtered)}  "
            f"missing: {len(missing)}  unexpected: {len(unexpected)}"
        )

    def forward(self, tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        pos = self.pos_embed(xyz)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        x = torch.cat([cls_tokens, tokens], dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)

        return x[:, 0]


class MinecraftPointBERTEncoder(nn.Module):
    """Voxel encoder lengkap: VoxelToPoints → InputAdapter → PointBERT → Head."""

    def __init__(
        self,
        num_points: int = 512,
        block_vocab_size: int = 256,
        block_embed_dim: int = 64,
        trans_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        pretrained_path: str = None,
        pretrained_type: str = "vanilla",
        use_fps_eval: bool = True,
    ):
        super().__init__()

        self.voxel_to_points = VoxelToPoints(
            num_points=num_points, use_fps_eval=use_fps_eval
        )
        self.block_embedding = nn.Embedding(block_vocab_size, block_embed_dim)

        in_channels = 3 + block_embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(in_channels, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, trans_dim),
        )

        self.backbone = PointBERTTransformer(
            trans_dim=trans_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )

        if pretrained_path is not None:
            if pretrained_type == "ulip":
                self.backbone.load_ulip_checkpoint(pretrained_path)
            else:
                self.backbone.load_point_bert_checkpoint(pretrained_path)

        self.freeze_backbone = freeze_backbone
        self._set_backbone_frozen(freeze_backbone)

        self.output_head = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, embed_dim),
        )
        self.block_embed_dim = block_embed_dim

    def _set_backbone_frozen(self, freeze: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def init_semantic_embeddings(
        self,
        index_to_name: list,
        sentence_model,
        freeze: bool = False,
        cache_dir: str = "checkpoints/cache",
    ):
        import os
        import hashlib
        import numpy as np

        vocab_size = self.block_embedding.num_embeddings
        embed_dim = self.block_embedding.embedding_dim

        names = list(index_to_name)[:vocab_size]
        while len(names) < vocab_size:
            names.append("<pad>")

        names_str = ",".join(names)
        vocab_hash = hashlib.md5(names_str.encode("utf-8")).hexdigest()
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"block_emb_{vocab_hash}.npy")

        raw_embs_np = None
        if os.path.exists(cache_path):
            print(
                f"[PointBERT] Strategy 2: Found cached embeddings at {cache_path}. Loading..."
            )
            try:
                raw_embs_np = np.load(cache_path)
            except Exception as e:
                print(
                    f"[PointBERT] Warning: Failed to load cache ({e}). Re-computing..."
                )

        if raw_embs_np is None:
            print(
                f"[PointBERT] Strategy 2: computing semantic embeddings for {vocab_size} block names..."
            )
            with torch.no_grad():
                raw_embs_tensor = sentence_model.encode(
                    names,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                    batch_size=256,
                )
                raw_embs_np = raw_embs_tensor.cpu().numpy()
                np.save(cache_path, raw_embs_np)

        device = self.block_embedding.weight.device
        raw_embs = torch.from_numpy(raw_embs_np).to(device)

        with torch.no_grad():
            text_dim = raw_embs.shape[-1]
            proj = nn.Linear(text_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            proj = proj.to(device)

            projected = proj(raw_embs)
            projected = F.normalize(projected, dim=-1)

            self.block_embedding.weight.data.copy_(projected)

        if freeze:
            self.block_embedding.weight.requires_grad = False
        else:
            self.block_embedding.weight.requires_grad = True

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        B = voxels.shape[0]
        M = self.voxel_to_points.num_points

        xyz, block_ids = self.voxel_to_points(voxels)
        block_feats = self.block_embedding(block_ids)

        combined = torch.cat([xyz, block_feats], dim=-1)
        combined_flat = combined.view(B * M, -1)
        tokens = self.input_projection(combined_flat)
        tokens = tokens.view(B, M, -1)

        global_feat = self.backbone(tokens, xyz)
        return self.output_head(global_feat)


# ---------------------------------------------------------------------------
# Text Encoder — Sentence Transformer + Projection
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Wraps a frozen sentence-transformer with a learned projection head."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        text_hidden_dim: int = 384,
        embed_dim: int = 256,
        freeze: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = SentenceTransformer(model_name)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(text_hidden_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode raw strings using the frozen sentence transformer."""
        # sentence-transformers returns numpy by default
        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B raw strings
        Returns:
            (B, embed_dim) embedding
        """
        with torch.no_grad():
            text_feats = self.encode_text(texts)  # (B, text_hidden_dim)
        # Move to same device as projection head, clone/detach from no_grad graph,
        # and re-enable grad so gradients flow through self.project during training.
        text_feats = (
            text_feats.to(next(self.project.parameters()).device)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        x = self.project(text_feats)  # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Dual Encoder — full model
# ---------------------------------------------------------------------------


class DualEncoder(nn.Module):
    """CLIP-style dual encoder wrapping text and voxel branches."""

    def __init__(self, cfg: dict, num_block_types: int):
        super().__init__()
        model_cfg = cfg["model"]
        dropout = model_cfg.get("dropout", 0.3)
        self.encoder_type = model_cfg.get("encoder_type", "cnn")

        if self.encoder_type == "pointbert":
            pb_cfg = cfg.get("pointbert", {})
            self.voxel_encoder = MinecraftPointBERTEncoder(
                num_points=pb_cfg.get("num_points", 512),
                block_vocab_size=num_block_types,
                block_embed_dim=pb_cfg.get("block_embed_dim", 64),
                trans_dim=pb_cfg.get("trans_dim", 384),
                depth=pb_cfg.get("depth", 12),
                num_heads=pb_cfg.get("num_heads", 6),
                mlp_ratio=pb_cfg.get("mlp_ratio", 4.0),
                drop_path=pb_cfg.get("drop_path", 0.1),
                embed_dim=model_cfg["embed_dim"],
                dropout=pb_cfg.get("dropout", 0.1),
                freeze_backbone=pb_cfg.get("freeze_backbone", True),
                pretrained_path=pb_cfg.get("pretrained_path", None),
                pretrained_type=pb_cfg.get("pretrained_type", "vanilla"),
                use_fps_eval=pb_cfg.get("use_fps_eval", True),
            )
        else:
            self.voxel_encoder = VoxelEncoder(
                num_block_types=num_block_types,
                block_embed_dim=model_cfg["block_embed_dim"],
                channels=model_cfg["voxel_channels"],
                embed_dim=model_cfg["embed_dim"],
                dropout=dropout,
                use_learned_stem=model_cfg.get("use_learned_stem", False),
                use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
            )

        self.text_encoder = TextEncoder(
            model_name=model_cfg["text_model"],
            text_hidden_dim=model_cfg["text_hidden_dim"],
            embed_dim=model_cfg["embed_dim"],
            freeze=model_cfg["freeze_text_encoder"],
            dropout=dropout,
        )

    def encode_voxel(self, voxels: torch.LongTensor) -> torch.Tensor:
        """Encode voxels and L2-normalise."""
        emb = self.voxel_encoder(voxels)
        return nn.functional.normalize(emb, dim=-1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text and L2-normalise."""
        emb = self.text_encoder(texts)
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, texts: list[str], voxels: torch.LongTensor):
        """
        Returns:
            text_emb: (B, embed_dim) normalised
            voxel_emb: (B, embed_dim) normalised
        """
        text_emb = self.encode_text(texts)
        voxel_emb = self.encode_voxel(voxels)
        return text_emb, voxel_emb

    def get_param_groups(self, cfg: dict) -> list:
        """Return parameter groups with custom learning rates if configured."""
        tr_cfg = cfg["training"]

        if self.encoder_type == "pointbert":
            pb_cfg = cfg.get("pointbert", {})
            lr_adapter = tr_cfg.get("lr_adapter", 3e-4)
            lr_head = tr_cfg.get("lr_head", 1e-4)
            lr_backbone = tr_cfg.get("lr_backbone", 5e-6)
            lr_text = tr_cfg.get("lr_text_proj", 1e-4)

            ve = self.voxel_encoder
            groups = [
                {
                    "params": list(ve.block_embedding.parameters()),
                    "lr": lr_adapter,
                    "name": "block_embed",
                },
                {
                    "params": list(ve.input_projection.parameters()),
                    "lr": lr_adapter,
                    "name": "input_proj",
                },
                {
                    "params": [ve.backbone.cls_token, ve.backbone.cls_pos],
                    "lr": lr_adapter,
                    "name": "cls_tokens",
                },
                {
                    "params": list(ve.backbone.pos_embed.parameters()),
                    "lr": lr_adapter,
                    "name": "pos_embed",
                },
                {
                    "params": list(ve.output_head.parameters()),
                    "lr": lr_head,
                    "name": "output_head",
                },
                {
                    "params": list(self.text_encoder.project.parameters()),
                    "lr": lr_text,
                    "name": "text_proj",
                },
            ]

            if not pb_cfg.get("freeze_backbone", True):
                groups.append(
                    {
                        "params": list(ve.backbone.blocks.parameters()),
                        "lr": lr_backbone,
                        "name": "transformer_blocks",
                    }
                )
                groups.append(
                    {
                        "params": list(ve.backbone.norm.parameters()),
                        "lr": lr_backbone,
                        "name": "transformer_norm",
                    }
                )
            return groups
        else:
            # Default CNN groupings
            return [
                {
                    "params": list(self.voxel_encoder.parameters()),
                    "lr": tr_cfg.get("lr_voxel", 1e-4),
                    "name": "voxel_encoder",
                },
                {
                    "params": list(self.text_encoder.project.parameters()),
                    "lr": tr_cfg.get("lr_text_proj", 1e-4),
                    "name": "text_proj",
                },
            ]

class TrimodalEncoder(nn.Module):
    """Trimodal encoder wrapping image, text (from TinyCLIP), and voxel branches."""

    def __init__(self, cfg: dict, num_block_types: int):
        super().__init__()
        model_cfg = cfg["model"]
        dropout = model_cfg.get("dropout", 0.3)
        self.encoder_type = model_cfg.get("encoder_type", "cnn")
        self.embed_dim = model_cfg["embed_dim"]

        # 1. Voxel Encoder
        if self.encoder_type == "pointbert":
            pb_cfg = cfg.get("pointbert", {})
            self.voxel_encoder = MinecraftPointBERTEncoder(
                num_points=pb_cfg.get("num_points", 512),
                block_vocab_size=num_block_types,
                block_embed_dim=pb_cfg.get("block_embed_dim", 64),
                trans_dim=pb_cfg.get("trans_dim", 384),
                depth=pb_cfg.get("depth", 12),
                num_heads=pb_cfg.get("num_heads", 6),
                mlp_ratio=pb_cfg.get("mlp_ratio", 4.0),
                drop_path=pb_cfg.get("drop_path", 0.1),
                embed_dim=self.embed_dim,
                dropout=pb_cfg.get("dropout", 0.1),
                freeze_backbone=pb_cfg.get("freeze_backbone", True),
                pretrained_path=pb_cfg.get("pretrained_path", None),
                pretrained_type=pb_cfg.get("pretrained_type", "vanilla"),
                use_fps_eval=pb_cfg.get("use_fps_eval", True),
            )
        else:
            self.voxel_encoder = VoxelEncoder(
                num_block_types=num_block_types,
                block_embed_dim=model_cfg["block_embed_dim"],
                channels=model_cfg["voxel_channels"],
                embed_dim=self.embed_dim,
                dropout=dropout,
                use_learned_stem=model_cfg.get("use_learned_stem", False),
                use_depthwise_separable=model_cfg.get("use_depthwise_separable", False),
            )

        # 2. Image and Text Encoders (from TinyCLIP)
        arch = model_cfg.get("tinyclip_arch", "TinyCLIP-auto-ViT-45M-32-Text-18M")
        pretrained = model_cfg.get("tinyclip_pretrained", "LAIONYFCC400M")
        print(f"[TrimodalEncoder] Loading TinyCLIP: {arch} ({pretrained})")
        
        self.arch = arch
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(arch)
        
        freeze_tinyclip = model_cfg.get("freeze_tinyclip", True)
        if freeze_tinyclip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
        # 3. Projection to embed_dim (e.g. 256)
        if hasattr(self.clip_model, "text_projection") and self.clip_model.text_projection is not None:
            clip_embed_dim = self.clip_model.text_projection.shape[1]
        elif hasattr(self.clip_model.visual, "output_dim"):
            clip_embed_dim = self.clip_model.visual.output_dim
        else:
            clip_embed_dim = 512
            
        self.clip_proj = nn.Linear(clip_embed_dim, self.embed_dim)

    def encode_voxel(self, voxels: torch.LongTensor) -> torch.Tensor:
        """Encode voxels and L2-normalise."""
        emb = self.voxel_encoder(voxels)
        return nn.functional.normalize(emb, dim=-1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text using TinyCLIP, project, and L2-normalise."""
        tokens = self.tokenizer(texts).to(next(self.clip_model.parameters()).device)
        requires_grad = any(p.requires_grad for p in self.clip_model.parameters())
        with torch.set_grad_enabled(requires_grad):
            emb = self.clip_model.encode_text(tokens)
            
        emb = self.clip_proj(emb)
        return nn.functional.normalize(emb, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image using TinyCLIP, project, and L2-normalise."""
        requires_grad = any(p.requires_grad for p in self.clip_model.parameters())
        with torch.set_grad_enabled(requires_grad):
            emb = self.clip_model.encode_image(images)
            
        emb = self.clip_proj(emb)
        return nn.functional.normalize(emb, dim=-1)

    def forward(self, texts: list[str], voxels: torch.LongTensor, images: torch.Tensor = None):
        """
        Returns:
            text_emb: (B, embed_dim) normalised
            voxel_emb: (B, embed_dim) normalised
            image_emb: (B, embed_dim) normalised (if images is provided)
        """
        text_emb = self.encode_text(texts)
        voxel_emb = self.encode_voxel(voxels)
        
        if images is not None:
            image_emb = self.encode_image(images)
            return text_emb, voxel_emb, image_emb
            
        return text_emb, voxel_emb

    def get_param_groups(self, cfg: dict) -> list:
        """Return parameter groups with custom learning rates if configured."""
        tr_cfg = cfg["training"]

        groups = []
        
        groups.append({
            "params": list(self.clip_proj.parameters()),
            "lr": tr_cfg.get("lr_text_proj", 1e-4),
            "name": "clip_proj",
        })
        
        if self.encoder_type == "pointbert":
            pb_cfg = cfg.get("pointbert", {})
            lr_adapter = tr_cfg.get("lr_adapter", 3e-4)
            lr_head = tr_cfg.get("lr_head", 1e-4)
            lr_backbone = tr_cfg.get("lr_backbone", 5e-6)

            ve = self.voxel_encoder
            groups.extend([
                {
                    "params": list(ve.block_embedding.parameters()),
                    "lr": lr_adapter,
                    "name": "block_embed",
                },
                {
                    "params": list(ve.input_projection.parameters()),
                    "lr": lr_adapter,
                    "name": "input_proj",
                },
                {
                    "params": [ve.backbone.cls_token, ve.backbone.cls_pos],
                    "lr": lr_adapter,
                    "name": "cls_tokens",
                },
                {
                    "params": list(ve.backbone.pos_embed.parameters()),
                    "lr": lr_adapter,
                    "name": "pos_embed",
                },
                {
                    "params": list(ve.output_head.parameters()),
                    "lr": lr_head,
                    "name": "output_head",
                },
            ])

            if not pb_cfg.get("freeze_backbone", True):
                groups.extend([
                    {
                        "params": list(ve.backbone.blocks.parameters()),
                        "lr": lr_backbone,
                        "name": "transformer_blocks",
                    },
                    {
                        "params": list(ve.backbone.norm.parameters()),
                        "lr": lr_backbone,
                        "name": "transformer_norm",
                    }
                ])
        else:
            groups.append({
                "params": list(self.voxel_encoder.parameters()),
                "lr": tr_cfg.get("lr_voxel", 1e-4),
                "name": "voxel_encoder",
            })
            
        if not cfg["model"].get("freeze_tinyclip", True):
            groups.append({
                "params": list(self.clip_model.parameters()),
                "lr": tr_cfg.get("lr_tinyclip", 1e-5),
                "name": "tinyclip",
            })

        return groups
