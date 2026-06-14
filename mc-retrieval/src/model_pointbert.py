"""Point-BERT VoxelEncoder — diimplementasikan sesuai kode asli Point-BERT.

Referensi: https://github.com/Julie-tang00/Point-BERT
Paper    : "Point-BERT: Pre-training 3D Point Cloud Transformers with
            Masked Point Modeling", CVPR 2022.

Arsitektur PERSIS dari Point_BERT.py aslinya:
  - Block     : Attention + Mlp + DropPath (bukan timm.Block, versi mereka sendiri)
  - TransformerEncoder: pos ditambah di SETIAP LAYER (x = block(x + pos))
  - pos_embed : MLP(3 → 128 → trans_dim)
  - cls_token + cls_pos: keduanya learnable parameter
  - Checkpoint format: base_ckpt dari 'base_model' key,
                       transformer_q.blocks.* → blocks.*

Pretrained weights:
  Download dari: https://github.com/Julie-tang00/Point-BERT (lihat README)
  Simpan di   : checkpoints/Point-BERT.pth
  Set di YAML : pointbert.pretrained_path: "checkpoints/Point-BERT.pth"

Pipeline:
  Voxel (B, 32, 32, 32)
    → VoxelToPoints       : non-air blocks → xyz (B,M,3) + block_ids (B,M)
    → InputAdapter        : block embed + MLP → tokens (B,M,384)
    → PointBERTTransformer: TransformerEncoder + pos_embed → CLS feat (B,384)
    → OutputHead          : MLP → embed_dim=256
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from model import TextEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Farthest Point Sampling (CPU / NumPy)
# ─────────────────────────────────────────────────────────────────────────────

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
    dist     = np.full(N, np.inf)
    farthest = np.random.randint(N)

    for i in range(n_samples):
        selected[i] = farthest
        d    = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Voxel Grid → Sparse Point Cloud
# ─────────────────────────────────────────────────────────────────────────────

class VoxelToPoints(nn.Module):
    """Konversi voxel 32³ → point cloud sparse berukuran tetap M.

    Per sample:
      1. Ambil posisi semua non-air block → koordinat (x, y, z)
      2. Zero-center + normalisasi ke ≈ [-1, 1]
      3. Sample tepat M titik
         · Training : random shuffle  (cepat + augmentasi stokastik)
         · Eval     : FPS             (deterministik + representatif)

    Returns:
        xyz:       (B, M, 3)  koordinat ternormalisasi
        block_ids: (B, M)     block type ID tiap titik
    """

    def __init__(self, num_points: int = 512, use_fps_eval: bool = True):
        super().__init__()
        self.num_points   = num_points
        self.use_fps_eval = use_fps_eval

    def _process_one(self, grid: torch.LongTensor):
        device = grid.device
        M      = self.num_points

        non_air = (grid != 0)
        coords  = non_air.nonzero(as_tuple=False).float()   # (N, 3)
        bids    = grid[non_air]                              # (N,)
        N       = len(coords)

        if N == 0:
            return (
                torch.zeros(M, 3, device=device),
                torch.zeros(M, dtype=torch.long, device=device),
            )

        # Zero-center + normalisasi [-1, 1]
        centroid = coords.mean(dim=0, keepdim=True)
        xyz      = coords - centroid
        scale    = xyz.abs().max().clamp(min=1.0)
        xyz      = xyz / scale

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
            idx  = torch.arange(N, device=device).repeat(reps)[:M]

        return xyz[idx], bids[idx]

    def forward(self, voxels: torch.LongTensor):
        """Args: voxels (B, 32, 32, 32) → xyz (B, M, 3), block_ids (B, M)"""
        xyz_list, bid_list = [], []
        for b in range(voxels.shape[0]):
            xyz, bids = self._process_one(voxels[b])
            xyz_list.append(xyz)
            bid_list.append(bids)
        return torch.stack(xyz_list), torch.stack(bid_list)


# ─────────────────────────────────────────────────────────────────────────────
# Point-BERT Building Blocks (SALIN PERSIS dari Point_BERT.py aslinya)
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    """MLP block — identik dengan Point_BERT.py line 17-33."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Self-attention block — identik dengan Point_BERT.py line 35-60."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block — identik dengan Point_BERT.py line 62-80."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder — identik dengan Point_BERT.py line 82-100.

    KUNCI: pos ditambahkan ke x DI SETIAP LAYER (bukan hanya di awal):
        for block in self.blocks:
            x = block(x + pos)
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list)
                          else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        # pos ditambah di setiap layer — PERSIS seperti kode asli mereka
        for block in self.blocks:
            x = block(x + pos)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stage D: Full Point-BERT Transformer (dengan pos_embed + cls_token + cls_pos)
# ─────────────────────────────────────────────────────────────────────────────

class PointBERTTransformer(nn.Module):
    """Transformer backbone dengan arsitektur PERSIS Point-BERT asli.

    Komponen yang sesuai dengan PointTransformer / MaskTransformer:
      - cls_token + cls_pos   : keduanya learnable
      - pos_embed             : MLP(3 → 128 → trans_dim)
      - blocks                : TransformerEncoder (pos tiap layer)
      - norm                  : LayerNorm di akhir

    Checkpoint loading:
      Kunci di checkpoint aslinya tersimpan sebagai:
        'base_model.transformer_q.blocks.blocks.0.norm1.weight'
      Setelah strip 'module.' dan ambil dari 'base_model':
        'transformer_q.blocks.blocks.0.norm1.weight'
      Kita remap ke struktur kita:
        'blocks.blocks.0.norm1.weight' → sudah sesuai TransformerEncoder.blocks
    """

    def __init__(
        self,
        trans_dim : int   = 384,
        depth     : int   = 12,
        num_heads : int   = 6,
        mlp_ratio : float = 4.0,
        drop_path : float = 0.1,
    ):
        super().__init__()
        self.trans_dim = trans_dim

        # Identik dengan PointTransformer.__init__ (line 124-141)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = TransformerEncoder(
            embed_dim      = trans_dim,
            depth          = depth,
            drop_path_rate = dpr,
            num_heads      = num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # Init — sama dengan _init_weights di MaskTransformer
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos,   std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    def load_point_bert_checkpoint(self, ckpt_path: str):
        """Load pretrained weights dari Point-BERT.pth checkpoint resmi.

        Format checkpoint asli (dari load_model_from_ckpt di Point_BERT.py):
          ckpt['base_model'] → strip 'module.' → strip 'transformer_q.'
          Hasilnya: 'blocks.blocks.0.*', 'norm.*', 'cls_token', 'cls_pos',
                    'pos_embed.*', 'encoder.*', 'reduce_dim.*'

        Kita ambil hanya bagian transformer (blocks, norm, cls, pos_embed).
        Bagian encoder (PointNet group encoder) kita skip karena
        kita pakai InputAdapter sendiri.
        """
        print(f"[PointBERT] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Step 1: ambil dari 'base_model', strip 'module.'
        if "base_model" in ckpt:
            raw = {k.replace("module.", ""): v
                   for k, v in ckpt["base_model"].items()}
        else:
            raw = {k.replace("module.", ""): v for k, v in ckpt.items()}

        # Step 2: ambil bagian transformer_q (encoder utama, bukan key encoder)
        # sesuai logic di load_model_from_ckpt line 181-185
        base = {}
        for k, v in raw.items():
            if k.startswith("transformer_q.") and not k.startswith("transformer_q.cls_head"):
                base[k[len("transformer_q."):]] = v
            elif k.startswith("base_model."):
                base[k[len("base_model."):]] = v

        if not base:
            # Fallback: coba langsung sebagai flat state dict
            base = raw

        # Step 3: filter hanya komponen yang ada di PointBERTTransformer kita
        # (skip encoder, reduce_dim, lm_head, cls_head, mask_token, dll.)
        keep_prefixes = (
            "blocks.",       # TransformerEncoder.blocks
            "norm.",         # LayerNorm akhir
            "cls_token",     # CLS token parameter
            "cls_pos",       # CLS pos parameter
            "pos_embed.",    # MLP pos embedding
        )
        filtered = {k: v for k, v in base.items()
                    if any(k.startswith(p) for p in keep_prefixes)
                    or k in ("cls_token", "cls_pos")}

        if not filtered:
            print("[PointBERT] WARNING: Tidak ada kunci yang cocok!")
            print(f"            Sample kunci tersedia: {list(base.keys())[:8]}")
            print("            Melanjutkan dengan random init.")
            return

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(f"[PointBERT] Checkpoint loaded — "
              f"matched: {loaded}/{len(filtered)}  "
              f"missing: {len(missing)}  unexpected: {len(unexpected)}")
        if missing:
            print(f"            Missing : {missing[:5]} ...")

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, trans_dim) — output dari InputAdapter
            xyz:    (B, N, 3)         — koordinat tiap titik
        Returns:
            (B, trans_dim) — CLS token feature (global representation)
        """
        B = tokens.shape[0]

        # Positional embedding dari koordinat 3D — SAMA dengan kode asli
        pos = self.pos_embed(xyz)           # (B, N, trans_dim)

        # Prepend CLS token + CLS pos
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, trans_dim)
        cls_pos    = self.cls_pos.expand(B, -1, -1)

        x   = torch.cat([cls_tokens, tokens], dim=1)    # (B, N+1, trans_dim)
        pos = torch.cat([cls_pos,    pos],    dim=1)    # (B, N+1, trans_dim)

        # TransformerEncoder: pos ditambah di setiap layer
        x = self.blocks(x, pos)   # (B, N+1, trans_dim)
        x = self.norm(x)

        return x[:, 0]   # CLS token → global feature (B, trans_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Full Voxel Encoder
# ─────────────────────────────────────────────────────────────────────────────

class MinecraftPointBERTEncoder(nn.Module):
    """Voxel encoder lengkap: VoxelToPoints → InputAdapter → PointBERT → Head."""

    def __init__(
        self,
        num_points       : int   = 512,
        block_vocab_size : int   = 256,
        block_embed_dim  : int   = 64,
        trans_dim        : int   = 384,
        depth            : int   = 12,
        num_heads        : int   = 6,
        mlp_ratio        : float = 4.0,
        drop_path        : float = 0.1,
        embed_dim        : int   = 256,
        dropout          : float = 0.1,
        freeze_backbone  : bool  = True,
        pretrained_path  : str   = None,
        use_fps_eval     : bool  = True,
    ):
        super().__init__()

        # ── Stage A: Voxel → Point Cloud ──────────────────────────────────
        self.voxel_to_points = VoxelToPoints(
            num_points   = num_points,
            use_fps_eval = use_fps_eval,
        )

        # ── Stage B: Block-ID Semantic Embedding ──────────────────────────
        self.block_embedding = nn.Embedding(block_vocab_size, block_embed_dim)

        # ── Stage C: Input Projection (domain adapter) ────────────────────
        # [xyz(3) ∥ block_feat(block_embed_dim)] → trans_dim(384)
        in_channels = 3 + block_embed_dim

        self.input_projection = nn.Sequential(
            nn.Linear(in_channels, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, trans_dim),
        )

        # ── Stage D: Point-BERT Transformer ───────────────────────────────
        self.backbone = PointBERTTransformer(
            trans_dim = trans_dim,
            depth     = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            drop_path = drop_path,
        )

        if pretrained_path is not None:
            self.backbone.load_point_bert_checkpoint(pretrained_path)
        else:
            print("[PointBERT] pretrained_path tidak diset → random init.")
            print("            Download Point-BERT.pth dari:")
            print("            https://github.com/Julie-tang00/Point-BERT")

        self.freeze_backbone = freeze_backbone
        self._set_backbone_frozen(freeze_backbone)

        # ── Stage E: Output Alignment Head ────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.LayerNorm(trans_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim, embed_dim),
        )

    def _set_backbone_frozen(self, freeze: bool):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        label = "❄️  FROZEN (Plan 2)" if freeze else "🔥 TRAINABLE (Plan 1)"
        print(f"[PointBERT] Backbone: {label}")

    def unfreeze_backbone(self):
        """Curriculum: panggil mid-training untuk switch Plan 2 → Plan 1."""
        self._set_backbone_frozen(False)
        self.freeze_backbone = False

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        B = voxels.shape[0]
        M = self.voxel_to_points.num_points

        # A: Voxel → Point Cloud
        xyz, block_ids = self.voxel_to_points(voxels)     # (B,M,3), (B,M)

        # B: Embed Block IDs
        block_feats = self.block_embedding(block_ids)      # (B, M, D_block)

        # C: Input Projection  X_i = [C_i ∥ F_i] → token
        combined      = torch.cat([xyz, block_feats], dim=-1)  # (B, M, 3+D)
        combined_flat = combined.view(B * M, -1)
        tokens        = self.input_projection(combined_flat)    # (B*M, trans_dim)
        tokens        = tokens.view(B, M, -1)                   # (B, M, trans_dim)

        # D: Point-BERT Transformer
        if self.freeze_backbone:
            with torch.no_grad():
                global_feat = self.backbone(tokens, xyz)   # (B, trans_dim)
        else:
            global_feat = self.backbone(tokens, xyz)

        # E: Output Head
        return self.output_head(global_feat)               # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# DualEncoder — drop-in replacement untuk DualEncoder di train.py
# ─────────────────────────────────────────────────────────────────────────────

class DualEncoderPointBERT(nn.Module):
    """CLIP-style dual encoder: Point-BERT voxel branch + TextEncoder."""

    def __init__(self, cfg: dict, num_block_types: int):
        super().__init__()

        model_cfg = cfg["model"]
        pb_cfg    = cfg.get("pointbert", {})

        self.voxel_encoder = MinecraftPointBERTEncoder(
            num_points       = pb_cfg.get("num_points",      512),
            block_vocab_size = num_block_types,
            block_embed_dim  = pb_cfg.get("block_embed_dim", 64),
            trans_dim        = pb_cfg.get("trans_dim",       384),
            depth            = pb_cfg.get("depth",           12),
            num_heads        = pb_cfg.get("num_heads",       6),
            mlp_ratio        = pb_cfg.get("mlp_ratio",       4.0),
            drop_path        = pb_cfg.get("drop_path",       0.1),
            embed_dim        = model_cfg["embed_dim"],
            dropout          = pb_cfg.get("dropout",         0.1),
            freeze_backbone  = pb_cfg.get("freeze_backbone", True),
            pretrained_path  = pb_cfg.get("pretrained_path", None),
            use_fps_eval     = pb_cfg.get("use_fps_eval",    True),
        )

        self.text_encoder = TextEncoder(
            model_name      = model_cfg["text_model"],
            text_hidden_dim = model_cfg["text_hidden_dim"],
            embed_dim       = model_cfg["embed_dim"],
            freeze          = model_cfg["freeze_text_encoder"],
            dropout         = model_cfg.get("dropout", 0.3),
        )

    def encode_voxel(self, voxels: torch.LongTensor) -> torch.Tensor:
        return F.normalize(self.voxel_encoder(voxels), dim=-1)

    def encode_text(self, texts: list) -> torch.Tensor:
        return F.normalize(self.text_encoder(texts), dim=-1)

    def forward(self, texts: list, voxels: torch.LongTensor):
        return self.encode_text(texts), self.encode_voxel(voxels)

    def param_groups(self, cfg: dict) -> list:
        """Discriminative LR groups untuk AdamW."""
        tr_cfg = cfg["training"]
        pb_cfg = cfg.get("pointbert", {})

        lr_adapter  = tr_cfg.get("lr_adapter",  3e-4)
        lr_head     = tr_cfg.get("lr_head",     1e-4)
        lr_backbone = tr_cfg.get("lr_backbone", 5e-6)
        lr_text     = tr_cfg.get("lr_text_proj",1e-4)

        ve = self.voxel_encoder
        groups = [
            # Input Adapter — selalu trainable
            {"params": ve.block_embedding.parameters(),  "lr": lr_adapter, "name": "block_embed"},
            {"params": ve.input_projection.parameters(), "lr": lr_adapter, "name": "input_proj"},
            # CLS token, CLS pos, pos_embed — selalu trainable
            {"params": [ve.backbone.cls_token,
                        ve.backbone.cls_pos],            "lr": lr_adapter, "name": "cls_tokens"},
            {"params": ve.backbone.pos_embed.parameters(),"lr": lr_adapter,"name": "pos_embed"},
            # Output head — selalu trainable
            {"params": ve.output_head.parameters(),      "lr": lr_head,    "name": "output_head"},
            # Text projection
            {"params": self.text_encoder.project.parameters(), "lr": lr_text, "name": "text_proj"},
        ]

        # Backbone TransformerEncoder blocks — hanya Plan 1
        if not pb_cfg.get("freeze_backbone", True):
            groups.append({
                "params": ve.backbone.blocks.parameters(),
                "lr"    : lr_backbone,
                "name"  : "transformer_blocks",
            })
            groups.append({
                "params": ve.backbone.norm.parameters(),
                "lr"    : lr_backbone,
                "name"  : "transformer_norm",
            })

        return groups
