from __future__ import annotations

from typing import List

import numpy as np


class FeatureQuantileTokenizer:
    """Feature-wise empirical quantile tokenizer.

    The tokenizer is fitted from observed entries only. Each feature has its
    own quantile bins and raw-scale bin representatives.
    """

    def __init__(self, n_bins: int = 32):
        self.n_bins = int(n_bins)
        self.edges_: List[np.ndarray] = []
        self.bin_representatives_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None

    def fit(self, X: np.ndarray, observed_mask: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if observed_mask is None:
            observed_mask = np.isfinite(X)
        observed_mask = np.asarray(observed_mask, dtype=bool) & np.isfinite(X)

        _, n_features = X.shape
        representatives = np.zeros((n_features, self.n_bins), dtype=np.float64)
        means = np.zeros(n_features, dtype=np.float64)
        edges_list: List[np.ndarray] = []

        for j in range(n_features):
            values = X[observed_mask[:, j], j]
            values = values[np.isfinite(values)]
            if values.size == 0:
                values = np.array([0.0], dtype=np.float64)

            means[j] = float(values.mean())
            edges = np.quantile(values, np.linspace(0.0, 1.0, self.n_bins + 1))

            for k in range(1, edges.size):
                if edges[k] <= edges[k - 1]:
                    edges[k] = edges[k - 1] + 1e-6

            bin_ids = np.searchsorted(edges[1:-1], values, side="right")
            bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)

            for k in range(self.n_bins):
                in_bin = values[bin_ids == k]
                representatives[j, k] = float(in_bin.mean()) if in_bin.size else means[j]

            edges_list.append(edges.astype(np.float64))

        self.edges_ = edges_list
        self.bin_representatives_ = representatives.astype(np.float32)
        self.feature_means_ = means.astype(np.float32)
        return self

    def transform_bins(self, X: np.ndarray) -> np.ndarray:
        if self.bin_representatives_ is None:
            raise RuntimeError("Tokenizer is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        bins = np.full(X.shape, -1, dtype=np.int64)
        for j, edges in enumerate(self.edges_):
            values = X[:, j]
            finite = np.isfinite(values)
            ids = np.searchsorted(edges[1:-1], values[finite], side="right")
            bins[finite, j] = np.clip(ids, 0, self.n_bins - 1)
        return bins

    def transform_quantile_midpoints(self, X: np.ndarray) -> np.ndarray:
        bins = self.transform_bins(X)
        out = np.full(bins.shape, np.nan, dtype=np.float32)
        finite = bins >= 0
        out[finite] = (bins[finite].astype(np.float32) + 0.5) / float(self.n_bins)
        return out

    def value_channel(self, X: np.ndarray) -> np.ndarray:
        """Map values to the image value channel range [-1, 1]."""
        q = self.transform_quantile_midpoints(X)
        return (2.0 * q - 1.0).astype(np.float32)
