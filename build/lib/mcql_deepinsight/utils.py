from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    d = str(device or "auto").lower()
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")
    if d == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def observed_col_mean_fill(X: np.ndarray):
    X = np.asarray(X, dtype=np.float32).copy()
    means = np.nanmean(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    rows, cols = np.where(~np.isfinite(X))
    X[rows, cols] = means[cols]
    return X.astype(np.float32), means
