from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def features_as_points(X_filled: np.ndarray) -> np.ndarray:
    """Return feature matrix with shape [n_features, n_samples]."""
    F = np.asarray(X_filled, dtype=np.float32).T
    F = F - F.mean(axis=1, keepdims=True)
    std = F.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (F / std).astype(np.float32)


def compute_feature_coordinates(
    X_filled: np.ndarray,
    reducer: str = "pca",
    random_state: int | None = 0,
    **reducer_params,
) -> np.ndarray:
    """Embed features into two-dimensional coordinates.

    Parameters
    ----------
    X_filled:
        Complete working matrix used only to learn the feature layout.
    reducer:
        ``"pca"`` by default. Optional ``"umap"`` requires ``umap-learn``.
    """
    F = features_as_points(X_filled)
    reducer = str(reducer).lower()

    if reducer == "pca":
        model = PCA(n_components=2, random_state=random_state)
        coords = model.fit_transform(F)
    elif reducer == "linear":
        coords = np.stack(
            [np.arange(F.shape[0], dtype=np.float32), np.zeros(F.shape[0], dtype=np.float32)],
            axis=1,
        )
    elif reducer == "umap":
        try:
            import umap
        except Exception as exc:
            raise ImportError("reducer='umap' requires umap-learn.") from exc
        params = {"n_components": 2, "random_state": random_state}
        params.update(reducer_params)
        coords = umap.UMAP(**params).fit_transform(F)
    else:
        raise KeyError(f"Unknown reducer={reducer!r}. Use 'pca', 'linear', or optional 'umap'.")

    return np.asarray(coords, dtype=np.float32)


def assign_pixels_rounded_overlap(coords: np.ndarray, image_size: int | tuple[int, int]):
    """Assign feature coordinates to image pixels with overlap allowed."""
    if isinstance(image_size, int):
        height = width = int(image_size)
    else:
        height, width = int(image_size[0]), int(image_size[1])

    coords = np.asarray(coords, dtype=np.float64)
    mins = coords.min(axis=0)
    spans = coords.max(axis=0) - mins
    spans = np.where(spans > 1e-12, spans, 1.0)
    normalized = (coords - mins) / spans

    rows = np.clip(np.rint(normalized[:, 1] * (height - 1)).astype(int), 0, height - 1)
    cols = np.clip(np.rint(normalized[:, 0] * (width - 1)).astype(int), 0, width - 1)
    feature_to_pixel = [(int(r), int(c)) for r, c in zip(rows, cols)]

    flat = rows * width + cols
    counts = np.bincount(flat, minlength=height * width)
    occupied = counts[counts > 0]
    audit = {
        "image_height": height,
        "image_width": width,
        "n_features": int(len(feature_to_pixel)),
        "n_unique_feature_pixels": int((counts > 0).sum()),
        "n_colliding_pixels": int((counts > 1).sum()),
        "n_features_in_collisions": int(counts[counts > 1].sum()) if (counts > 1).any() else 0,
        "max_features_per_pixel": int(occupied.max()) if occupied.size else 0,
        "mean_features_per_occupied_pixel": float(occupied.mean()) if occupied.size else 0.0,
    }
    return feature_to_pixel, audit
