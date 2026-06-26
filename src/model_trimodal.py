"""Trimodal encoder: flexible Text + Image + Voxel encoders.

Text encoder options  (model.text_encoder):
  "clip"    — CLIP text model (frozen) + learnable projection
  "minilm"  — all-MiniLM-L6-v2 (frozen) + learnable projection

Image encoder options (model.image_encoder):
  "clip"    — CLIP ViT vision model (frozen) + learnable projection
              handles multi-view: input (B, N, 3, H, W) → project → mean-pool → (B, D)

Projection options (model.proj_type):
  "linear"  — single Linear layer (default, original)
  "mlp"     — two-layer MLP with GELU; for image: project-per-view then average (like TriCoLo)

Voxel encoder (pointbert section):
  pretrained_type: "vanilla" | "ulip"
  freeze_backbone: true | false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared projection builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_proj(in_dim: int, out_dim: int, proj_type: str) -> nn.Module:
    """Linear (default) or 2-layer MLP with GELU."""
    if proj_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
    return nn.Linear(in_dim, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Text encoders
# ─────────────────────────────────────────────────────────────────────────────

class CLIPTextEncoder(nn.Module):
    """CLIP text backbone (frozen) + learnable projection → embed_dim.

    Accepts either:
      - list[str]      — raw text, runs full CLIP pipeline
      - torch.Tensor   — pre-cached CLIP features (B, clip_out_dim), skips backbone
    """

    def __init__(self, cfg: dict, processor):
        super().__init__()
        from transformers import CLIPModel
        clip_name = cfg["model"].get("clip_model", "openai/clip-vit-base-patch16")
        clip = CLIPModel.from_pretrained(clip_name)
        self.text_model = clip.text_model
        self.text_proj = clip.text_projection
        self.processor = processor
        self.max_length = 77

        if cfg["model"].get("freeze_text_encoder", True):
            for p in self.text_model.parameters():
                p.requires_grad = False
            for p in self.text_proj.parameters():
                p.requires_grad = False

        clip_out_dim = cfg["model"].get("clip_output_dim", 512)
        proj_type = cfg["model"].get("proj_type", "linear")
        self.proj = _build_proj(clip_out_dim, cfg["model"]["embed_dim"], proj_type)

    def forward(self, texts) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            # cached path: (B, clip_out_dim) — skip frozen backbone
            return F.normalize(self.proj(texts), dim=-1)
        device = next(self.parameters()).device
        inputs = self.processor(
            text=texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self.text_model(**inputs)
        feats = self.text_proj(out.pooler_output)
        return F.normalize(self.proj(feats), dim=-1)

    def backbone_params(self):
        return list(self.text_model.parameters()) + list(self.text_proj.parameters())


class MiniLMTextEncoder(nn.Module):
    """all-MiniLM-L6-v2 (frozen) + learnable projection → embed_dim."""

    def __init__(self, cfg: dict):
        super().__init__()
        from model import TextEncoder
        self.encoder = TextEncoder(cfg)

        if cfg["model"].get("freeze_text_encoder", True):
            for p in self.encoder.parameters():
                p.requires_grad = False

        text_hidden = cfg["model"].get("text_hidden_dim", 384)
        proj_type = cfg["model"].get("proj_type", "linear")
        self.proj = _build_proj(text_hidden, cfg["model"]["embed_dim"], proj_type)

    def forward(self, texts) -> torch.Tensor:
        feats = self.encoder(texts)
        return F.normalize(self.proj(feats), dim=-1)

    def backbone_params(self):
        return list(self.encoder.parameters())


def build_text_encoder(cfg: dict, processor=None) -> nn.Module:
    enc = cfg["model"].get("text_encoder", "clip")
    if enc == "clip":
        assert processor is not None, "CLIPTextEncoder requires a CLIPProcessor"
        return CLIPTextEncoder(cfg, processor)
    elif enc == "minilm":
        return MiniLMTextEncoder(cfg)
    else:
        raise ValueError(f"Unknown text_encoder: '{enc}'. Choose 'clip' or 'minilm'.")


# ─────────────────────────────────────────────────────────────────────────────
# Image encoders
# ─────────────────────────────────────────────────────────────────────────────

class CLIPImageEncoder(nn.Module):
    """CLIP vision backbone (frozen) + learnable projection → embed_dim.

    proj_type="linear": average views → project (original)
    proj_type="mlp":    project each view → average (TriCoLo-style, more expressive)

    Accepts:
      - (B, N_views, 3, H, W)     — raw pixels, runs full CLIP pipeline
      - (B, N_views, clip_out_dim) — pre-cached features, skips frozen backbone
    """

    def __init__(self, cfg: dict):
        super().__init__()
        from transformers import CLIPModel
        clip_name = cfg["model"].get("clip_model", "openai/clip-vit-base-patch16")
        clip = CLIPModel.from_pretrained(clip_name)
        self.vision_model = clip.vision_model
        self.visual_proj = clip.visual_projection

        if cfg["model"].get("freeze_image_encoder", True):
            for p in self.vision_model.parameters():
                p.requires_grad = False
            for p in self.visual_proj.parameters():
                p.requires_grad = False

        clip_out_dim = cfg["model"].get("clip_output_dim", 512)
        proj_type = cfg["model"].get("proj_type", "linear")
        self.proj = _build_proj(clip_out_dim, cfg["model"]["embed_dim"], proj_type)
        self.proj_type = proj_type

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 3:
            # cached path: (B, N_views, clip_out_dim)
            if self.proj_type == "mlp":
                B, N, D = pixel_values.shape
                feats = self.proj(pixel_values.view(B * N, D))  # project each view
                feats = feats.view(B, N, -1).mean(dim=1)        # then average
            else:
                feats = self.proj(pixel_values.mean(dim=1))     # average then project
        else:
            # raw path: (B, N_views, 3, H, W)
            B, N, C, H, W = pixel_values.shape
            pv = pixel_values.view(B * N, C, H, W)
            out = self.vision_model(pixel_values=pv)
            raw = self.visual_proj(out.pooler_output)           # (B*N, clip_out_dim)
            if self.proj_type == "mlp":
                feats = self.proj(raw).view(B, N, -1).mean(dim=1)  # project then average
            else:
                feats = self.proj(raw.view(B, N, -1).mean(dim=1))  # average then project
        return F.normalize(feats, dim=-1)

    def backbone_params(self):
        return list(self.vision_model.parameters()) + list(self.visual_proj.parameters())


def build_image_encoder(cfg: dict) -> nn.Module:
    enc = cfg["model"].get("image_encoder", "clip")
    if enc == "clip":
        return CLIPImageEncoder(cfg)
    else:
        raise ValueError(f"Unknown image_encoder: '{enc}'. Currently supported: 'clip'.")


# ─────────────────────────────────────────────────────────────────────────────
# Voxel encoder
# ─────────────────────────────────────────────────────────────────────────────

def build_voxel_encoder(cfg: dict, num_block_types: int) -> nn.Module:
    from model import MinecraftPointBERTEncoder
    pb = cfg.get("pointbert", {})
    m = cfg["model"]
    return MinecraftPointBERTEncoder(
        num_points=pb.get("num_points", 1024),
        block_vocab_size=num_block_types,
        block_embed_dim=pb.get("block_embed_dim", 128),
        trans_dim=pb.get("trans_dim", 384),
        depth=pb.get("depth", 12),
        num_heads=pb.get("num_heads", 6),
        mlp_ratio=pb.get("mlp_ratio", 4.0),
        drop_path=pb.get("drop_path", 0.1),
        embed_dim=m["embed_dim"],
        dropout=pb.get("dropout", 0.1),
        freeze_backbone=pb.get("freeze_backbone", True),
        pretrained_path=pb.get("pretrained_path"),
        pretrained_type=pb.get("pretrained_type", "vanilla"),
        use_fps_eval=pb.get("use_fps_eval", True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trimodal encoder
# ─────────────────────────────────────────────────────────────────────────────

class TriModalEncoder(nn.Module):
    """Unified trimodal encoder: text + image + voxel → shared embed_dim space."""

    def __init__(self, cfg: dict, num_block_types: int, processor=None):
        super().__init__()
        self.text_encoder = build_text_encoder(cfg, processor)
        self.image_encoder = build_image_encoder(cfg)
        self.voxel_encoder = build_voxel_encoder(cfg, num_block_types)

    def encode_text(self, texts) -> torch.Tensor:
        return self.text_encoder(texts)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)

    def encode_voxel(self, voxels: torch.LongTensor) -> torch.Tensor:
        return F.normalize(self.voxel_encoder(voxels), dim=-1)

    def forward(self, texts, pixel_values, voxels):
        return self.encode_text(texts), self.encode_image(pixel_values), self.encode_voxel(voxels)

    def get_param_groups(self, cfg: dict) -> list:
        tr = cfg["training"]
        pb = cfg.get("pointbert", {})
        ve = self.voxel_encoder

        groups = [
            {
                "name": "text_proj",
                "params": list(self.text_encoder.proj.parameters()),
                "lr": tr.get("lr_text_proj", 1e-4),
            },
            {
                "name": "image_proj",
                "params": list(self.image_encoder.proj.parameters()),
                "lr": tr.get("lr_image_proj", 1e-4),
            },
            {
                "name": "pb_adapter",
                "params": list(ve.block_embedding.parameters()) + list(ve.input_projection.parameters()),
                "lr": tr.get("lr_adapter", 3e-4),
            },
            {
                "name": "pb_head",
                "params": list(ve.output_head.parameters()),
                "lr": tr.get("lr_head", 1e-4),
            },
        ]

        if not cfg["model"].get("freeze_text_encoder", True):
            groups.append({
                "name": "text_backbone",
                "params": self.text_encoder.backbone_params(),
                "lr": tr.get("lr_text_backbone", 1e-5),
            })

        if not cfg["model"].get("freeze_image_encoder", True):
            groups.append({
                "name": "image_backbone",
                "params": self.image_encoder.backbone_params(),
                "lr": tr.get("lr_image_backbone", 1e-5),
            })

        if not pb.get("freeze_backbone", True):
            groups.append({
                "name": "pb_backbone",
                "params": list(ve.backbone.parameters()),
                "lr": tr.get("lr_backbone", 5e-6),
            })

        return groups
