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


class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR."""
    def __init__(self, temperature_init: float = 0.1):
        super().__init__()
        self.temperature = temperature_init

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (B, D) L2-normalised embeddings
        Returns:
            scalar loss
        """
        B = z1.size(0)
        # Concat views: (2B, D)
        z = torch.cat([z1, z2], dim=0)
        
        # Sim matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature
        
        # Labels for positive pairs: 
        # z1[i] is positive with z2[i], i.e., index i with i+B, and i+B with i
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -float("inf"))
        
        loss = F.cross_entropy(sim, labels)
        return loss

class NTXentLoss(nn.Module):
    """
    This NTXentLoss implementation is adapted from TriCoLo:
    https://github.com/edreisMD/ConVIRT-pytorch/blob/master/loss/nt_xent.py
    """
    def __init__(self, temperature=0.07, alpha_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight

    def _softXEnt(self, target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        batch_size = zis.shape[0]
        labels = torch.eye(batch_size, device=zis.device, dtype=torch.float32)

        logits_ab = torch.matmul(zis, torch.transpose(zjs, 0, 1)) / self.temperature
        logits_ba = torch.matmul(zjs, torch.transpose(zis, 0, 1)) / self.temperature

        loss_a = self._softXEnt(labels, logits_ab)
        loss_b = self._softXEnt(labels, logits_ba)

        return self.alpha_weight * loss_a + (1 - self.alpha_weight) * loss_b

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _pairwise_distances(self, zis, zls, squared=False):
        dot_product = torch.matmul(zls, zis.t())
        a_square_norm = torch.diag(torch.matmul(zls, zls.t()))
        b_square_norm = torch.diag(torch.matmul(zis, zis.t()))
        distances = a_square_norm.unsqueeze(0) - 2.0 * dot_product + b_square_norm.unsqueeze(1)
        distances[distances < 0] = 0
        if not squared:
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16
            distances = (1.0 - mask) * torch.sqrt(distances)
        return distances

    def forward(self, zis, zls):
        batch_size = zis.shape[0]
        distances = self._pairwise_distances(zis, zls)
        loss_list = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                if distances[i][i] < distances[i][j] < distances[i][i] + self.margin:  # semi-hard
                    loss_list.append(distances[i][i] - distances[i][j] + self.margin)

        if len(loss_list) == 0:  # margin is set to a too small value
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:
                        continue
                    if distances[i][j] < distances[i][i]:  # hard
                        loss_list.append(distances[i][i] - distances[i][j] + self.margin)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=zis.device)

        loss = sum(loss_list) / len(loss_list)
        return loss
