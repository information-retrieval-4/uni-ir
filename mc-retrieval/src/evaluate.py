"""Retrieval evaluation: Recall@k, MRR, Median Rank."""

import argparse

import torch
import numpy as np
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder
from losses import CLIPLoss
from utils import load_config, set_seed, get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Run model on all batches and collect embeddings.

    Returns:
        text_embs:  (N, D) tensor
        voxel_embs: (N, D) tensor
        all_texts:  list of N strings
    """
    model.eval()
    text_embs, voxel_embs = [], []
    all_texts = []

    for texts, voxels in tqdm(loader, desc="Encoding", leave=False):
        voxels = voxels.to(device)
        t_emb = model.encode_text(texts)
        v_emb = model.encode_voxel(voxels)
        text_embs.append(t_emb.cpu())
        voxel_embs.append(v_emb.cpu())
        all_texts.extend(texts)

    text_embs = torch.cat(text_embs, dim=0)
    voxel_embs = torch.cat(voxel_embs, dim=0)
    return text_embs, voxel_embs, all_texts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """Compute retrieval metrics assuming query[i] matches gallery[i].

    Args:
        query_embs:   (N, D)
        gallery_embs: (N, D)
        ks: list of k values for Recall@k

    Returns:
        dict with recall@k, mrr, median_rank
    """
    # cosine similarity (embeddings should already be L2-normed)
    sim = query_embs @ gallery_embs.T          # (N, N)
    N = sim.shape[0]

    # ranks of the ground-truth match (diagonal)
    # for each query i, rank of gallery i
    ranks = []
    for i in range(N):
        scores = sim[i]
        # number of items with higher similarity than the correct one
        rank = (scores > scores[i]).sum().item() + 1   # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = float((ranks <= k).mean())
    metrics["mrr"] = float((1.0 / ranks).mean())
    metrics["median_rank"] = float(np.median(ranks))
    metrics["mean_rank"] = float(np.mean(ranks))

    return metrics


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: dict, checkpoint_path: str = None):
    """Run full retrieval evaluation on the test set."""
    set_seed(cfg["data"]["seed"])
    device = get_device()

    # load checkpoint
    if checkpoint_path is None:
        checkpoint_path = f"{cfg['training']['checkpoint_dir']}/best.pt"
    ckpt = load_checkpoint(checkpoint_path, device)

    # rebuild data loaders
    _, _, test_loader, block_mapping, num_blocks = create_dataloaders(cfg)

    # rebuild model
    model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
    model.load_state_dict(ckpt["model_state"])

    # extract embeddings
    text_embs, voxel_embs, texts = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(text_embs)} embeddings (dim={text_embs.shape[1]})")

    ks = cfg["eval"]["recall_k"]

    # text → voxel retrieval
    t2v = compute_retrieval_metrics(text_embs, voxel_embs, ks=ks)
    print("\n=== Text → Voxel Retrieval ===")
    for k, v in t2v.items():
        print(f"  {k}: {v:.4f}")

    # voxel → text retrieval
    v2t = compute_retrieval_metrics(voxel_embs, text_embs, ks=ks)
    print("\n=== Voxel → Text Retrieval ===")
    for k, v in v2t.items():
        print(f"  {k}: {v:.4f}")

    return {"text_to_voxel": t2v, "voxel_to_text": v2t, "texts": texts}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)
