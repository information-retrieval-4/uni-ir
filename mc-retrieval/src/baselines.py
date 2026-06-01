"""Baseline evaluation: Random, Text-only, BM25, Voxel-only retrieval."""

import argparse
import re
import json

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from dataset import create_dataloaders, build_text, clean_text
from model import DualEncoder
from evaluate import compute_retrieval_metrics, compute_category_metrics
from utils import load_config, set_seed, get_device, load_checkpoint


def _self_retrieval_metrics(embs, categories, ks):
    """Compute retrieval metrics for same-modality retrieval (excludes self-match)."""
    sim = embs @ embs.T
    # mask out diagonal so items can't retrieve themselves
    sim.fill_diagonal_(-float('inf'))
    N = sim.shape[0]

    ranks = []
    for i in range(N):
        scores = sim[i]
        # ground truth is still index i, but it's masked out
        # so we measure: among non-self items, how does the most similar
        # item rank? For same-modality there's no single "correct" answer,
        # so we use category-level metrics only for instance-level
        rank = (scores > scores[i]).sum().item() + 1  # will be N since diag=-inf
        ranks.append(rank)
    ranks = np.array(ranks)

    # instance metrics don't make sense for same-modality (no GT pair)
    # but we report them as N/A placeholder
    instance = {}
    for k in ks:
        instance[f"recall@{k}"] = float('nan')
    instance["mrr"] = float('nan')
    instance["median_rank"] = float('nan')
    instance["mean_rank"] = float('nan')

    # category metrics: does the top-k contain same-category items?
    sorted_indices = torch.argsort(sim, dim=1, descending=True).numpy()
    q_cats = np.array(categories)

    category = {}
    for k in ks:
        precisions = []
        hit_rates = []
        for i in range(N):
            top_k_idx = sorted_indices[i, :k]
            n_relevant = (q_cats[top_k_idx] == q_cats[i]).sum()
            precisions.append(n_relevant / k)
            hit_rates.append(1.0 if n_relevant > 0 else 0.0)
        category[f"cat_precision@{k}"] = float(np.mean(precisions))
        category[f"cat_hit_rate@{k}"] = float(np.mean(hit_rates))

    return instance, category


def random_baseline(n: int, ks: list[int]) -> dict:
    """Analytical random retrieval baseline."""
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = k / n
    metrics["mrr"] = sum(1.0 / i for i in range(1, n + 1)) / n
    metrics["median_rank"] = n / 2
    metrics["mean_rank"] = (n + 1) / 2
    return metrics


def text_only_baseline(test_loader, device, ks: list[int]):
    """Text-only retrieval using frozen sentence-transformer cosine similarity.

    Uses the raw sentence-transformer embeddings (no projection, no training).
    Retrieval = query text → find most similar text → return its paired voxel.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.eval()

    all_texts = []
    all_categories = []
    for texts, voxels, categories in test_loader:
        all_texts.extend(texts)
        all_categories.extend(categories)

    # encode all texts
    with torch.no_grad():
        text_embs = model.encode(all_texts, convert_to_tensor=True,
                                 show_progress_bar=False)
        text_embs = torch.nn.functional.normalize(text_embs, dim=-1).cpu()

    # text→text similarity (excluding self-match)
    return _self_retrieval_metrics(text_embs, all_categories, ks)




def bm25_baseline(test_loader, ks: list[int]):
    """BM25/TF-IDF text retrieval baseline."""
    all_texts = []
    all_categories = []
    for texts, voxels, categories in test_loader:
        all_texts.extend(texts)
        all_categories.extend(categories)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(all_texts)

    from sklearn.preprocessing import normalize
    tfidf_norm = normalize(tfidf, norm="l2")
    sim = torch.from_numpy((tfidf_norm @ tfidf_norm.T).toarray()).float()
    sim.fill_diagonal_(-float('inf'))  # exclude self-match

    N = len(all_texts)
    sorted_indices = torch.argsort(sim, dim=1, descending=True).numpy()
    q_cats = np.array(all_categories)

    # instance-level: N/A for same-modality
    instance = {}
    for k in ks:
        instance[f"recall@{k}"] = float('nan')
    instance["mrr"] = float('nan')
    instance["median_rank"] = float('nan')
    instance["mean_rank"] = float('nan')

    category = {}
    for k in ks:
        precisions = []
        hit_rates = []
        for i in range(N):
            top_k_idx = sorted_indices[i, :k]
            n_relevant = (q_cats[top_k_idx] == q_cats[i]).sum()
            precisions.append(n_relevant / k)
            hit_rates.append(1.0 if n_relevant > 0 else 0.0)
        category[f"cat_precision@{k}"] = float(np.mean(precisions))
        category[f"cat_hit_rate@{k}"] = float(np.mean(hit_rates))

    return instance, category


@torch.no_grad()
def voxel_only_baseline(test_loader, model, device, ks: list[int]):
    """Voxel-only retrieval using trained voxel encoder embeddings."""
    model.eval()
    voxel_embs = []
    all_categories = []

    for texts, voxels, categories in tqdm(test_loader, desc="Encoding voxels", leave=False):
        voxels = voxels.to(device)
        v_emb = model.encode_voxel(voxels)
        voxel_embs.append(v_emb.cpu())
        all_categories.extend(categories)

    voxel_embs = torch.cat(voxel_embs, dim=0)

    return _self_retrieval_metrics(voxel_embs, all_categories, ks)


def run_baselines(cfg: dict, checkpoint_path: str = None):
    """Run all baselines and print comparison."""
    set_seed(cfg["data"]["seed"])
    device = get_device()
    ks = cfg["eval"]["recall_k"]

    # load data
    _, _, test_loader, block_mapping, num_blocks = create_dataloaders(cfg)
    N = sum(len(texts) for texts, _, _ in test_loader)
    print(f"Test set size: {N}")

    # --- Random ---
    print("\n" + "=" * 60)
    print("RANDOM BASELINE")
    print("=" * 60)
    rand = random_baseline(N, ks)
    for k, v in rand.items():
        print(f"  {k}: {v:.4f}")

    # --- BM25/TF-IDF ---
    print("\n" + "=" * 60)
    print("BM25 (TF-IDF) TEXT BASELINE")
    print("=" * 60)
    bm25_inst, bm25_cat = bm25_baseline(test_loader, ks)
    print("\n  Instance-level:")
    for k, v in bm25_inst.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Category-level:")
    for k, v in bm25_cat.items():
        print(f"    {k}: {v:.4f}")

    # --- Text-only (dense) ---
    print("\n" + "=" * 60)
    print("TEXT-ONLY (SENTENCE-TRANSFORMER) BASELINE")
    print("=" * 60)
    text_inst, text_cat = text_only_baseline(test_loader, device, ks)
    print("\n  Instance-level:")
    for k, v in text_inst.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Category-level:")
    for k, v in text_cat.items():
        print(f"    {k}: {v:.4f}")

    # --- Voxel-only ---
    if checkpoint_path:
        print("\n" + "=" * 60)
        print("VOXEL-ONLY BASELINE (trained encoder)")
        print("=" * 60)
        ckpt = load_checkpoint(checkpoint_path, device)
        model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
        model.load_state_dict(ckpt["model_state"])
        vox_inst, vox_cat = voxel_only_baseline(test_loader, model, device, ks)
        print("\n  Instance-level:")
        for k, v in vox_inst.items():
            print(f"    {k}: {v:.4f}")
        print("\n  Category-level:")
        for k, v in vox_cat.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for voxel-only baseline (best model)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_baselines(cfg, args.checkpoint)
