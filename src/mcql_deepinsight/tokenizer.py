from __future__ import annotations

from typing import List

import numpy as np


class FeatureQuantileTokenizer:
    """Feature-wise quantile tokenizer used by MCQL.

    Each feature has its own empirical quantile bins and raw-scale bin
    representatives. The tokenizer is fit only on observed entries.
    """

    def __init__(self, n_bins: int = 32):
        self.n_bins = int(n_bins)
        self.edges_: List[np.ndarray] = []
        self.bin_representatives_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None

    def fit(self, X: np.ndarray, observed_mask: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if observed_mask is None:
            observed_mask = np.isfinite(X)
        observed_mask = np.asarray(observed_mask, dtype=bool) & np.isfinite(X)

        n, p = X.shape
        reps = np.zeros((p, self.n_bins), dtype=np.float64)
        means = np.zeros(p, dtype=np.float64)
        edges_list: List[np.ndarray] = []

        for j in range(p):
            vals = X[observed_mask[:, j], j]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                vals = np.array([0.0], dtype=np.float64)
            means[j] = float(np.mean(vals))

            edges = np.quantile(vals, np.linspace(0, 1, self.n_bins + 1))
            for k in range(1, edges.size):
                if edges[k] <= edges[k - 1]:
                    edges[k] = edges[k - 1] + 1e-6
            edges_list.append(edges.astype(np.float64))

            bins = np.searchsorted(edges[1:-1], vals, side="right")
            bins = np.clip(bins, 0, self.n_bins - 1)
            for k in range(self.n_bins):
                vv = vals[bins == k]
                reps[j, k] = float(np.mean(vv)) if vv.size else means[j]

        self.edges_ = edges_list
        self.bin_representatives_ = reps.astype(np.float32)
        self.feature_means_ = means.astype(np.float32)
        return self

    def transform_bins(self, X: np.ndarray) -> np.ndarray:
        if self.bin_representatives_ is None:
            raise RuntimeError("Tokenizer is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape, -1, dtype=np.int64)
        for j, edges in enumerate(self.edges_):
            vals = X[:, j]
            finite = np.isfinite(vals)
            bins = np.searchsorted(edges[1:-1], vals[finite], side="right")
            out[finite, j] = np.clip(bins, 0, self.n_bins - 1)
        return out

    def transform_u(self, X: np.ndarray) -> np.ndarray:
        bins = self.transform_bins(X)
        out = np.full(bins.shape, np.nan, dtype=np.float32)
        finite = bins >= 0
        out[finite] = (bins[finite].astype(np.float32) + 0.5) / float(self.n_bins)
        return out

    def value_channel(self, X: np.ndarray) -> np.ndarray:
        """Map observed values to [-1, 1] quantile value channel."""
        U = self.transform_u(X)
        return (2.0 * U - 1.0).astype(np.float32)
