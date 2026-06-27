"""Utility helpers: config loading, seeding, device, checkpointing."""

import os
import random
import yaml
import torch
import numpy as np


def load_config(path: str = "configs/cnn/cnn_default.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: torch.device = None) -> dict:
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    state = torch.load(path, map_location=device, weights_only=False)
    print(f"Checkpoint loaded from {path}")
    return state
