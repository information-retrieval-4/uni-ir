"""Retrieval evaluation: Recall@k, MRR, Median Rank + Category-level metrics."""

import argparse

import torch
import numpy as np
from tqdm import tqdm

from dataset import create_dataloaders
from model import DualEncoder, TrimodalEncoder
from utils import load_config, set_seed, get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    text_embs, voxel_embs, image_embs = [], [], []
    all_texts = []
    all_categories = []

    for batch in tqdm(loader, desc="Encoding", leave=False):
        if len(batch) == 4:
            texts, voxels, images, categories = batch
            images = images.to(device)
        else:
            texts, voxels, categories = batch
            images = None

        voxels = voxels.to(device)
        
        t_emb = model.encode_text(texts)
        v_emb = model.encode_voxel(voxels)
        
        text_embs.append(t_emb.cpu())
        voxel_embs.append(v_emb.cpu())
        
        if images is not None:
            i_emb = model.encode_image(images)
            image_embs.append(i_emb.cpu())
            
        all_texts.extend(texts)
        all_categories.extend(categories)

    text_embs = torch.cat(text_embs, dim=0)
    voxel_embs = torch.cat(voxel_embs, dim=0)
    if len(image_embs) > 0:
        image_embs = torch.cat(image_embs, dim=0)
    else:
        image_embs = None
        
    return text_embs, voxel_embs, image_embs, all_texts, all_categories


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


def compute_category_metrics(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_categories: list[str],
    gallery_categories: list[str],
    ks: list[int] = [1, 5, 10],
) -> dict:
    """Compute category-level retrieval metrics.

    A retrieved item is 'relevant' if it shares the same category as the query.

    Returns:
        dict with category_precision@k, category_recall@k, and category_hit_rate@k
    """
    sim = query_embs @ gallery_embs.T          # (N, N)
    N = sim.shape[0]

    # precompute category arrays
    q_cats = np.array(query_categories)
    g_cats = np.array(gallery_categories)

    # for each query, get sorted indices of gallery by descending similarity
    sorted_indices = torch.argsort(sim, dim=1, descending=True).numpy()

    metrics = {}
    for k in ks:
        precisions = []
        hit_rates = []
        for i in range(N):
            top_k_idx = sorted_indices[i, :k]
            top_k_cats = g_cats[top_k_idx]
            query_cat = q_cats[i]

            # how many of top-k share the category?
            n_relevant = (top_k_cats == query_cat).sum()
            precisions.append(n_relevant / k)
            hit_rates.append(1.0 if n_relevant > 0 else 0.0)

        metrics[f"cat_precision@{k}"] = float(np.mean(precisions))
        metrics[f"cat_hit_rate@{k}"] = float(np.mean(hit_rates))

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

    # rebuild model
    use_trimodal = cfg.get("model", {}).get("use_trimodal", False)
    # create dataloaders needs max_block_types which is accessible via data cfg
    num_blocks = cfg["data"]["max_block_types"]
    
    if use_trimodal:
        model = TrimodalEncoder(cfg, num_block_types=num_blocks).to(device)
        image_preprocess = getattr(model, "preprocess", None)
    else:
        model = DualEncoder(cfg, num_block_types=num_blocks).to(device)
        image_preprocess = None
        
    model.load_state_dict(ckpt["model_state"])

    # rebuild data loaders
    _, _, test_loader, block_mapping, num_blocks, _ = create_dataloaders(cfg, image_preprocess=image_preprocess)

    # extract embeddings
    text_embs, voxel_embs, image_embs, texts, categories = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(text_embs)} embeddings (dim={text_embs.shape[1]})")

    # show category distribution
    unique_cats, cat_counts = np.unique(categories, return_counts=True)
    print(f"\nCategories in test set: {len(unique_cats)}")
    for cat, cnt in sorted(zip(unique_cats, cat_counts), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {cnt}")

    ks = cfg["eval"]["recall_k"]

    # --- instance-level ---
    print("\n" + "=" * 50)
    print("INSTANCE-LEVEL RETRIEVAL")
    print("=" * 50)

    t2v = compute_retrieval_metrics(text_embs, voxel_embs, ks=ks)
    print("\n  Text → Voxel:")
    for k, v in t2v.items():
        print(f"    {k}: {v:.4f}")

    v2t = compute_retrieval_metrics(voxel_embs, text_embs, ks=ks)
    print("\n  Voxel → Text:")
    for k, v in v2t.items():
        print(f"    {k}: {v:.4f}")

    if image_embs is not None:
        i2v = compute_retrieval_metrics(image_embs, voxel_embs, ks=ks)
        print("\n  Image → Voxel:")
        for k, v in i2v.items():
            print(f"    {k}: {v:.4f}")

        v2i = compute_retrieval_metrics(voxel_embs, image_embs, ks=ks)
        print("\n  Voxel → Image:")
        for k, v in v2i.items():
            print(f"    {k}: {v:.4f}")
            
        t2i = compute_retrieval_metrics(text_embs, image_embs, ks=ks)
        print("\n  Text → Image:")
        for k, v in t2i.items():
            print(f"    {k}: {v:.4f}")

    # --- category-level ---
    print("\n" + "=" * 50)
    print("CATEGORY-LEVEL RETRIEVAL")
    print("=" * 50)

    t2v_cat = compute_category_metrics(text_embs, voxel_embs, categories, categories, ks=ks)
    print("\n  Text → Voxel (category match):")
    for k, v in t2v_cat.items():
        print(f"    {k}: {v:.4f}")

    v2t_cat = compute_category_metrics(voxel_embs, text_embs, categories, categories, ks=ks)
    print("\n  Voxel → Text (category match):")
    for k, v in v2t_cat.items():
        print(f"    {k}: {v:.4f}")
        
    if image_embs is not None:
        i2v_cat = compute_category_metrics(image_embs, voxel_embs, categories, categories, ks=ks)
        print("\n  Image → Voxel (category match):")
        for k, v in i2v_cat.items():
            print(f"    {k}: {v:.4f}")

        v2i_cat = compute_category_metrics(voxel_embs, image_embs, categories, categories, ks=ks)
        print("\n  Voxel → Image (category match):")
        for k, v in v2i_cat.items():
            print(f"    {k}: {v:.4f}")
            
        t2i_cat = compute_category_metrics(text_embs, image_embs, categories, categories, ks=ks)
        print("\n  Text → Image (category match):")
        for k, v in t2i_cat.items():
            print(f"    {k}: {v:.4f}")

    res = {
        "text_to_voxel": t2v,
        "voxel_to_text": v2t,
        "text_to_voxel_cat": t2v_cat,
        "voxel_to_text_cat": v2t_cat,
        "texts": texts,
        "categories": categories,
    }
    
    if image_embs is not None:
        res["image_to_voxel"] = i2v
        res["voxel_to_image"] = v2i
        res["text_to_image"] = t2i
        res["image_to_voxel_cat"] = i2v_cat
        res["voxel_to_image_cat"] = v2i_cat
        res["text_to_image_cat"] = t2i_cat

    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval model")
    parser.add_argument("--config", type=str, default="configs/cnn_default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint)
