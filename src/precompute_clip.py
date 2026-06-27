import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, set_seed, get_device
from dataset import create_dataloaders
from model import TrimodalEncoder

@torch.no_grad()
def run_split(clip_model, dataloader, device, split_name, save_path):
    clip_model.eval()
    all_embeddings = []
    
    print(f"Precomputing {split_name} split...")
    for batch in tqdm(dataloader):
        if len(batch) == 4:
            texts, voxels, images, categories = batch
            images = images.to(device)
            
            # images shape: (B, 3, 224, 224) or (B, V, 3, 224, 224)
            is_multiview = images.ndim == 5
            if is_multiview:
                B, V, C, H, W = images.shape
                images = images.view(B * V, C, H, W)
                
            # encode using frozen clip
            with torch.amp.autocast(device.type) if hasattr(torch, "amp") else torch.autocast(device.type):
                emb = clip_model.encode_image(images)
            
            if is_multiview:
                emb = emb.view(B, V, -1).mean(dim=1)
                
            all_embeddings.append(emb.cpu())
        else:
            raise ValueError("Precomputation requires images, but dataset did not return them.")
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[{split_name}] Extracted embeddings shape: {all_embeddings.shape}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_embeddings, save_path)
    print(f"[{split_name}] Saved to {save_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/trimodal/trimodal_tinyclip.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["use_cached_clip"] = False
    set_seed(cfg["training"].get("seed", 42))
    device = get_device()

    # Load full model just to get the initialized TinyCLIP backbone and preprocess
    model = TrimodalEncoder(cfg, num_block_types=256).to(device)
    clip_model = model.clip_model
    image_preprocess = model.preprocess

    # Get dataloaders
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(
        cfg, image_preprocess=image_preprocess, load_voxels=False
    )

    # We need unshuffled versions for caching deterministically matching the dataset index!
    def make_unshuffled(loader):
        return DataLoader(
            loader.dataset,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            collate_fn=loader.collate_fn,
            pin_memory=True
        )

    unshuffled_train = make_unshuffled(train_loader)
    unshuffled_val = make_unshuffled(val_loader)
    unshuffled_test = make_unshuffled(test_loader)

    cache_dir = "data/clip_cache"
    run_split(clip_model, unshuffled_train, device, "train", f"{cache_dir}/train.pt")
    run_split(clip_model, unshuffled_val, device, "val", f"{cache_dir}/val.pt")
    run_split(clip_model, unshuffled_test, device, "test", f"{cache_dir}/test.pt")

if __name__ == "__main__":
    main()
