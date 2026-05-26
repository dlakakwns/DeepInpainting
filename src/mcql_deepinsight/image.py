from __future__ import annotations

import numpy as np


def build_pixel_groups(feature_to_pixel, image_size):
    if isinstance(image_size, int):
        H = W = int(image_size)
    else:
        H, W = int(image_size[0]), int(image_size[1])
    feature_to_flat = np.asarray([int(r) * W + int(c) for r, c in feature_to_pixel], dtype=np.int64)
    occupied_flat = np.unique(feature_to_flat)
    occupied_mask_flat = np.zeros(H * W, dtype=bool)
    occupied_mask_flat[occupied_flat] = True
    feature_rows = np.asarray([int(r) for r, c in feature_to_pixel], dtype=np.int64)
    feature_cols = np.asarray([int(c) for r, c in feature_to_pixel], dtype=np.int64)
    return feature_to_flat, occupied_mask_flat, feature_rows, feature_cols


def aggregate_feature_values_to_image(
    V_features: np.ndarray,
    unavailable_features: np.ndarray,
    feature_to_flat: np.ndarray,
    occupied_mask_flat: np.ndarray,
    image_size: int | tuple[int, int],
):
    if isinstance(image_size, int):
        H = W = int(image_size)
    else:
        H, W = int(image_size[0]), int(image_size[1])

    single = V_features.ndim == 1
    if single:
        V_features = V_features[None, :]
        unavailable_features = unavailable_features[None, :]

    n, p = V_features.shape
    out = np.full((n, H * W), -3.0, dtype=np.float32)

    for i in range(n):
        valid = ~unavailable_features[i]
        sums = np.zeros(H * W, dtype=np.float32)
        counts = np.zeros(H * W, dtype=np.float32)
        if valid.any():
            np.add.at(sums, feature_to_flat[valid], V_features[i, valid].astype(np.float32))
            np.add.at(counts, feature_to_flat[valid], 1.0)
        out[i, occupied_mask_flat] = -2.0
        has_valid = counts > 0
        out[i, has_valid] = sums[has_valid] / counts[has_valid]

    out = out.reshape(n, H, W)
    return out[0] if single else out


def coordinate_channels(image_size: int | tuple[int, int]):
    if isinstance(image_size, int):
        H = W = int(image_size)
    else:
        H, W = int(image_size[0]), int(image_size[1])
    coord_x = np.tile(np.linspace(-1, 1, W, dtype=np.float32)[None, :], (H, 1))
    coord_y = np.tile(np.linspace(-1, 1, H, dtype=np.float32)[:, None], (1, W))
    return coord_x, coord_y
