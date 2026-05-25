"""CLIP-style symmetric contrastive loss (InfoNCE)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """Symmetric InfoNCE loss with a learnable temperature.

    Given L2-normalised text embeddings T and voxel embeddings V of shape
    (B, D), computes:
        logits = (T @ V^T) / τ          — (B, B) cosine-similarity matrix
        loss   = (CE(logits, labels) + CE(logits^T, labels)) / 2

    where labels = [0, 1, ..., B-1] (each sample matches itself).
    """

    def __init__(self, temperature_init: float = 0.07):
        super().__init__()
        # learnable log-temperature (clamped for stability)
        self.log_temp = nn.Parameter(torch.tensor(temperature_init).log())

    @property
    def temperature(self) -> torch.Tensor:
        # clamp between ~0.01 and ~1.0
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        text_emb: torch.Tensor,
        voxel_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:  (B, D) L2-normalised text embeddings
            voxel_emb: (B, D) L2-normalised voxel embeddings
        Returns:
            scalar loss
        """
        # cosine similarity scaled by temperature
        logits = (text_emb @ voxel_emb.T) / self.temperature   # (B, B)

        labels = torch.arange(len(logits), device=logits.device)

        loss_t2v = F.cross_entropy(logits, labels)        # text  → voxel
        loss_v2t = F.cross_entropy(logits.T, labels)      # voxel → text

        return (loss_t2v + loss_v2t) / 2.0
