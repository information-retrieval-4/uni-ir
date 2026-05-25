"""Model architectures: VoxelEncoder, TextEncoder, DualEncoder."""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


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
    ):
        super().__init__()

        self.block_embedding = nn.Embedding(num_block_types, block_embed_dim)

        layers = []
        in_ch = block_embed_dim
        for out_ch in channels[:-1]:
            layers.extend([
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.GELU(),
                nn.MaxPool3d(2),
            ])
            in_ch = out_ch

        # last conv block uses adaptive pooling instead of maxpool
        layers.extend([
            nn.Conv3d(in_ch, channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[-1]),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        ])

        self.conv_stack = nn.Sequential(*layers)
        self.project = nn.Linear(channels[-1], embed_dim)

    def forward(self, voxels: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            voxels: (B, 32, 32, 32) block ID tensor
        Returns:
            (B, embed_dim) L2-normalised embedding
        """
        x = self.block_embedding(voxels)           # (B, 32, 32, 32, D)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, 32, 32, 32)
        x = self.conv_stack(x)                       # (B, C, 1, 1, 1)
        x = x.flatten(1)                             # (B, C)
        x = self.project(x)                          # (B, embed_dim)
        return x


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
    ):
        super().__init__()

        self.encoder = SentenceTransformer(model_name)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.project = nn.Sequential(
            nn.Linear(text_hidden_dim, embed_dim),
            nn.GELU(),
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
        text_feats = text_feats.to(next(self.project.parameters()).device)
        x = self.project(text_feats)               # (B, embed_dim)
        return x


# ---------------------------------------------------------------------------
# Dual Encoder — full model
# ---------------------------------------------------------------------------

class DualEncoder(nn.Module):
    """CLIP-style dual encoder wrapping text and voxel branches."""

    def __init__(self, cfg: dict, num_block_types: int):
        super().__init__()
        model_cfg = cfg["model"]

        self.voxel_encoder = VoxelEncoder(
            num_block_types=num_block_types,
            block_embed_dim=model_cfg["block_embed_dim"],
            channels=model_cfg["voxel_channels"],
            embed_dim=model_cfg["embed_dim"],
        )
        self.text_encoder = TextEncoder(
            model_name=model_cfg["text_model"],
            text_hidden_dim=model_cfg["text_hidden_dim"],
            embed_dim=model_cfg["embed_dim"],
            freeze=model_cfg["freeze_text_encoder"],
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
