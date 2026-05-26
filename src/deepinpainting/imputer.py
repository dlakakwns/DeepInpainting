from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .image import aggregate_feature_values_to_image, build_pixel_groups, coordinate_channels
from .layout import assign_pixels_rounded_overlap, compute_feature_coordinates
from .model import DeepInpaintingNet, count_parameters
from .tokenizer import FeatureQuantileTokenizer
from .utils import observed_column_mean_fill, resolve_device, seed_everything


class _DeepInpaintingDataset(Dataset):
    def __init__(
        self,
        state: Dict[str, Any],
        pseudo_mask_rate: float,
        indices,
        fixed_seed: Optional[int] = None,
    ):
        self.state = state
        self.pseudo_mask_rate = float(pseudo_mask_rate)
        self.indices = np.asarray(indices, dtype=int)
        self.fixed_pseudo_masks = None

        if fixed_seed is not None:
            rng = np.random.default_rng(int(fixed_seed))
            self.fixed_pseudo_masks = []
            target_bins = state["target_bins"]
            actual_missing = state["actual_missing_features"]
            for row_index in self.indices:
                eligible = (~actual_missing[int(row_index)]) & (target_bins[int(row_index)] >= 0)
                pseudo = (rng.random(eligible.shape) < self.pseudo_mask_rate) & eligible
                if pseudo.sum() == 0 and eligible.sum() > 0:
                    pseudo[rng.choice(np.flatnonzero(eligible))] = True
                self.fixed_pseudo_masks.append(pseudo.astype(bool))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_index = int(self.indices[int(index)])
        state = self.state
        actual_missing = state["actual_missing_features"][row_index].copy()
        target_bins = state["target_bins"]

        if self.fixed_pseudo_masks is None:
            eligible = (~actual_missing) & (target_bins[row_index] >= 0)
            pseudo = (np.random.random(eligible.shape) < self.pseudo_mask_rate) & eligible
            if pseudo.sum() == 0 and eligible.sum() > 0:
                pseudo[np.random.choice(np.flatnonzero(eligible))] = True
        else:
            pseudo = self.fixed_pseudo_masks[int(index)].copy()

        unavailable = actual_missing | pseudo
        value_image = aggregate_feature_values_to_image(
            state["value_features"][row_index],
            unavailable,
            state["feature_to_flat"],
            state["occupied_mask_flat"],
            state["image_size"],
        )
        image = np.stack([value_image, state["coord_x"], state["coord_y"]], axis=0).astype(np.float32)

        return (
            torch.from_numpy(image),
            torch.from_numpy(target_bins[row_index]).long(),
            torch.from_numpy(pseudo).bool(),
        )


def _masked_quantile_loss(logits, target_bins, pseudo_mask, feature_rows, feature_cols):
    if pseudo_mask.sum() == 0:
        return logits.sum() * 0.0

    target_on_loss = target_bins[pseudo_mask]
    if torch.any(target_on_loss < 0) or torch.any(target_on_loss >= logits.shape[1]):
        raise RuntimeError("Actual missing target sentinel entered the training loss.")

    # [batch, K, H, W] -> [batch, n_features, K]
    feature_logits = logits[:, :, feature_rows, feature_cols].permute(0, 2, 1)
    return F.cross_entropy(feature_logits[pseudo_mask], target_on_loss)


@torch.no_grad()
def _evaluate_epoch(model, loader, device, feature_rows, feature_cols):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0

    for image, target_bins, pseudo_mask in loader:
        image = image.to(device)
        target_bins = target_bins.to(device)
        pseudo_mask = pseudo_mask.to(device)

        logits = model(image)
        loss = _masked_quantile_loss(logits, target_bins, pseudo_mask, feature_rows, feature_cols)
        n_targets = int(pseudo_mask.sum().item())

        total_loss += float(loss.detach().cpu()) * max(n_targets, 1)
        total_count += n_targets

        if n_targets > 0:
            predicted_bins = logits[:, :, feature_rows, feature_cols].argmax(dim=1)
            correct += int((predicted_bins[pseudo_mask] == target_bins[pseudo_mask]).sum().item())

    return {
        "validation_loss": total_loss / max(total_count, 1),
        "validation_targets": int(total_count),
        "validation_accuracy": correct / max(total_count, 1),
    }


class DeepInpaintingImputer:
    """DeepInpainting estimator for incomplete tabular matrices.

    Missing values must be encoded as ``np.nan``. The estimator preserves
    observed entries exactly in the transformed output.
    """

    def __init__(
        self,
        n_bins: int = 32,
        image_size: int = 64,
        reducer: str = "pca",
        reducer_params: Optional[Dict[str, Any]] = None,
        hidden_channels: int = 64,
        num_blocks: int = 7,
        dropout: float = 0.05,
        pseudo_mask_rate: float = 0.05,
        validation_pseudo_mask_rate: float = 0.05,
        batch_size: int = 8,
        eval_batch_size: int = 16,
        max_epochs: int = 80,
        patience: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = "auto",
        random_state: Optional[int] = 0,
        torch_num_threads: Optional[int] = 1,
        verbose: bool = False,
    ):
        self.n_bins = int(n_bins)
        self.image_size = int(image_size)
        self.reducer = reducer
        self.reducer_params = dict(reducer_params or {})
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.pseudo_mask_rate = float(pseudo_mask_rate)
        self.validation_pseudo_mask_rate = float(validation_pseudo_mask_rate)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = str(device)
        self.random_state = random_state
        self.torch_num_threads = torch_num_threads
        self.verbose = bool(verbose)

    def _set_torch_threads(self):
        if self.torch_num_threads is not None:
            torch.set_num_threads(int(self.torch_num_threads))

    def _check_is_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("DeepInpaintingImputer is not fitted.")

    def _build_training_state(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        actual_missing = ~np.isfinite(X)
        X_filled, column_means = observed_column_mean_fill(X)

        tokenizer = FeatureQuantileTokenizer(self.n_bins).fit(X, np.isfinite(X))
        coords = compute_feature_coordinates(
            X_filled,
            reducer=self.reducer,
            random_state=self.random_state,
            **self.reducer_params,
        )
        feature_to_pixel, layout_audit = assign_pixels_rounded_overlap(coords, self.image_size)
        feature_to_flat, occupied_mask_flat, feature_rows, feature_cols = build_pixel_groups(feature_to_pixel, self.image_size)
        coord_x, coord_y = coordinate_channels(self.image_size)

        value_features = tokenizer.value_channel(X_filled).astype(np.float32)
        target_bins = tokenizer.transform_bins(X_filled).astype(np.int64)
        target_bins[actual_missing] = -1

        state = {
            "value_features": value_features,
            "target_bins": target_bins,
            "actual_missing_features": actual_missing.astype(bool),
            "feature_to_flat": feature_to_flat,
            "occupied_mask_flat": occupied_mask_flat,
            "feature_rows": feature_rows,
            "feature_cols": feature_cols,
            "coord_x": coord_x,
            "coord_y": coord_y,
            "image_size": int(self.image_size),
        }
        return state, tokenizer, feature_to_pixel, coords, layout_audit, column_means

    def fit(self, X, y=None):
        self._set_torch_threads()
        seed_everything(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        state, tokenizer, feature_to_pixel, coords, layout_audit, column_means = self._build_training_state(X)

        self.n_features_in_ = int(X.shape[1])
        self.column_means_ = column_means.astype(np.float32)
        self.tokenizer_ = tokenizer
        self.feature_to_pixel_ = feature_to_pixel
        self.feature_coordinates_ = coords
        self.layout_audit_ = layout_audit
        self._state_ = state

        indices = np.arange(X.shape[0])
        split_seed = (0 if self.random_state is None else int(self.random_state) + 100) % (2**32 - 1)
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=0.25,
            random_state=split_seed,
            stratify=None,
        )

        validation_seed = (0 if self.random_state is None else int(self.random_state) + 300) % (2**32 - 1)
        train_dataset = _DeepInpaintingDataset(state, self.pseudo_mask_rate, train_indices)
        validation_dataset = _DeepInpaintingDataset(
            state,
            self.validation_pseudo_mask_rate,
            validation_indices,
            fixed_seed=validation_seed,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, max(len(train_dataset), 1)),
            shuffle=True,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(self.eval_batch_size, max(len(validation_dataset), 1)),
            shuffle=False,
            num_workers=0,
        )

        device = resolve_device(self.device)
        feature_rows = torch.from_numpy(state["feature_rows"]).long().to(device)
        feature_cols = torch.from_numpy(state["feature_cols"]).long().to(device)

        model = DeepInpaintingNet(
            n_bins=self.n_bins,
            hidden_channels=self.hidden_channels,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_validation_loss = float("inf")
        best_epoch = 0
        patience_left = int(self.patience)
        history = []

        epoch_iter = range(1, self.max_epochs + 1)
        if self.verbose:
            epoch_iter = tqdm(epoch_iter, desc="DeepInpainting epochs", unit="epoch")

        for epoch in epoch_iter:
            model.train()
            train_loss_sum = 0.0
            train_target_count = 0

            for image, target_bins, pseudo_mask in train_loader:
                image = image.to(device)
                target_bins = target_bins.to(device)
                pseudo_mask = pseudo_mask.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss = _masked_quantile_loss(logits, target_bins, pseudo_mask, feature_rows, feature_cols)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                n_targets = int(pseudo_mask.sum().item())
                train_loss_sum += float(loss.detach().cpu()) * max(n_targets, 1)
                train_target_count += n_targets

            validation = _evaluate_epoch(model, validation_loader, device, feature_rows, feature_cols)
            row = {
                "epoch": int(epoch),
                "training_loss": train_loss_sum / max(train_target_count, 1),
                "training_targets": int(train_target_count),
                **validation,
                "n_parameters": int(count_parameters(model)),
            }
            history.append(row)

            if validation["validation_loss"] < best_validation_loss - 1e-5:
                best_validation_loss = float(validation["validation_loss"])
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                patience_left = int(self.patience)
            else:
                patience_left -= 1

            if patience_left <= 0:
                break

        model.load_state_dict(best_state)

        self.model_ = model
        self.training_history_ = history
        self.best_epoch_ = int(best_epoch)
        self.best_validation_loss_ = float(best_validation_loss)
        return self

    def _build_inference_images(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected X with shape [n_samples, {self.n_features_in_}].")

        missing = ~np.isfinite(X)
        X_filled = X.copy()
        rows, cols = np.where(~np.isfinite(X_filled))
        X_filled[rows, cols] = self.column_means_[cols]

        value_features = self.tokenizer_.value_channel(X_filled).astype(np.float32)
        feature_to_flat, occupied_mask_flat, _, _ = build_pixel_groups(self.feature_to_pixel_, self.image_size)
        coord_x, coord_y = coordinate_channels(self.image_size)

        base_images = aggregate_feature_values_to_image(
            value_features,
            missing,
            feature_to_flat,
            occupied_mask_flat,
            self.image_size,
        ).astype(np.float32)

        return base_images, coord_x, coord_y, missing

    @torch.no_grad()
    def transform(self, X):
        self._check_is_fitted()
        self._set_torch_threads()

        X = np.asarray(X, dtype=np.float32)
        base_images, coord_x, coord_y, missing = self._build_inference_images(X)

        completed = X.copy()
        rows, cols = np.where(~np.isfinite(completed))
        completed[rows, cols] = self.column_means_[cols]

        if not missing.any():
            return completed.astype(np.float32)

        device = resolve_device(self.device)
        self.model_.eval()

        representatives = self.tokenizer_.bin_representatives_.astype(np.float32)
        feature_rows = np.asarray([r for r, _ in self.feature_to_pixel_], dtype=int)
        feature_cols = np.asarray([c for _, c in self.feature_to_pixel_], dtype=int)

        for start in range(0, X.shape[0], int(self.eval_batch_size)):
            end = min(X.shape[0], start + int(self.eval_batch_size))
            value_tensor = torch.from_numpy(base_images[start:end]).float()
            xcoord_tensor = torch.from_numpy(np.repeat(coord_x[None, ...], end - start, axis=0)).float()
            ycoord_tensor = torch.from_numpy(np.repeat(coord_y[None, ...], end - start, axis=0)).float()
            image = torch.stack([value_tensor, xcoord_tensor, ycoord_tensor], dim=1).to(device)

            logits = self.model_(image).detach().cpu().numpy()
            logits = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

            batch_missing = missing[start:end]
            for j, (r, c) in enumerate(zip(feature_rows, feature_cols)):
                rows_missing = batch_missing[:, j]
                if not rows_missing.any():
                    continue
                probability_vectors = probabilities[rows_missing, :, int(r), int(c)]
                decoded_values = probability_vectors @ representatives[j]
                completed[start:end, j][rows_missing] = decoded_values.astype(np.float32)

        completed[~missing] = X[~missing]
        return completed.astype(np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)
