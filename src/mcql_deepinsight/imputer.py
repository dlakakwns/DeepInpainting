from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .image import aggregate_feature_values_to_image, build_pixel_groups, coordinate_channels
from .layout import assign_pixels_rounded_overlap, compute_feature_coordinates
from .model import GlobalPoolMCQLNet, count_parameters
from .tokenizer import FeatureQuantileTokenizer
from .utils import get_device, observed_col_mean_fill, seed_everything


class _MCQLDataset(Dataset):
    def __init__(self, state: Dict[str, Any], pseudo_rate: float, indices, fixed_seed: Optional[int] = None):
        self.state = state
        self.pseudo_rate = float(pseudo_rate)
        self.indices = np.asarray(indices, dtype=int)
        self.fixed = None
        if fixed_seed is not None:
            rng = np.random.default_rng(int(fixed_seed))
            self.fixed = []
            B = state["B_features"]
            actual = state["actual_missing_features"]
            for i in self.indices:
                eligible = (~actual[int(i)]) & (B[int(i)] >= 0)
                pseudo = (rng.random(eligible.shape) < self.pseudo_rate) & eligible
                if pseudo.sum() == 0 and eligible.sum() > 0:
                    pseudo[rng.choice(np.flatnonzero(eligible))] = True
                self.fixed.append(pseudo.astype(bool))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[int(idx)])
        state = self.state
        actual = state["actual_missing_features"][i].copy()
        B = state["B_features"]

        if self.fixed is None:
            eligible = (~actual) & (B[i] >= 0)
            pseudo = (np.random.random(eligible.shape) < self.pseudo_rate) & eligible
            if pseudo.sum() == 0 and eligible.sum() > 0:
                pseudo[np.random.choice(np.flatnonzero(eligible))] = True
        else:
            pseudo = self.fixed[int(idx)].copy()

        unavailable = actual | pseudo
        V = aggregate_feature_values_to_image(
            state["V_features"][i],
            unavailable,
            state["feature_to_flat"],
            state["occupied_mask_flat"],
            state["image_size"],
        )
        inp = np.stack([V, state["coord_x"], state["coord_y"]], axis=0).astype(np.float32)
        return (
            torch.from_numpy(inp),
            torch.from_numpy(B[i]).long(),
            torch.from_numpy(pseudo).bool(),
        )


def _mcql_loss(logits, target_bins_features, pseudo_feature_mask, feature_rows, feature_cols):
    if pseudo_feature_mask.sum() == 0:
        return logits.sum() * 0.0

    target_on_loss = target_bins_features[pseudo_feature_mask]
    if torch.any(target_on_loss < 0) or torch.any(target_on_loss >= logits.shape[1]):
        raise RuntimeError("Actual missing target sentinel entered MCQL loss.")

    feature_logits = logits[:, :, feature_rows, feature_cols].permute(0, 2, 1)
    return F.cross_entropy(feature_logits[pseudo_feature_mask], target_on_loss)


@torch.no_grad()
def _evaluate_model(model, loader, device, feature_rows, feature_cols):
    model.eval()
    total, count, correct = 0.0, 0, 0
    for inp, target, pseudo in loader:
        inp = inp.to(device)
        target = target.to(device)
        pseudo = pseudo.to(device)
        logits = model(inp)
        loss = _mcql_loss(logits, target, pseudo, feature_rows, feature_cols)
        n = int(pseudo.sum().item())
        total += float(loss.detach().cpu()) * max(n, 1)
        count += n
        if n > 0:
            pred = logits[:, :, feature_rows, feature_cols].argmax(dim=1)
            correct += int((pred[pseudo] == target[pseudo]).sum().item())
    return {
        "val_mcql": total / max(count, 1),
        "val_pseudo_count": int(count),
        "val_acc": correct / max(count, 1),
    }


class MCQLImputer:
    """Method-only DeepInsight MCQL imputer.

    Parameters are intentionally focused on the proposed method. Benchmark
    backends are not included in this estimator.

    The estimator expects a 2D numeric matrix with missing values encoded as
    ``np.nan``.
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
        val_pseudo_mask_rate: float = 0.05,
        batch_size: int = 8,
        eval_batch_size: int = 16,
        max_epochs: int = 80,
        patience: int = 10,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        device: str = "auto",
        random_state: Optional[int] = 0,
        verbose: bool = False,
        torch_num_threads: Optional[int] = 1,
    ):
        self.n_bins = int(n_bins)
        self.image_size = int(image_size)
        self.reducer = reducer
        self.reducer_params = dict(reducer_params or {})
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)
        self.pseudo_mask_rate = float(pseudo_mask_rate)
        self.val_pseudo_mask_rate = float(val_pseudo_mask_rate)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = str(device)
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.torch_num_threads = torch_num_threads

    def _check_is_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("MCQLImputer is not fitted.")

    def _build_training_state(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        M = ~np.isfinite(X)
        X_filled, col_means = observed_col_mean_fill(X)

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

        V_features = tokenizer.value_channel(X_filled).astype(np.float32)
        B_features = tokenizer.transform_bins(X_filled).astype(np.int64)
        B_features[M] = -1

        state = {
            "V_features": V_features,
            "B_features": B_features,
            "actual_missing_features": M.astype(bool),
            "feature_to_flat": feature_to_flat,
            "occupied_mask_flat": occupied_mask_flat,
            "feature_rows": feature_rows,
            "feature_cols": feature_cols,
            "coord_x": coord_x,
            "coord_y": coord_y,
            "image_size": int(self.image_size),
        }
        return state, tokenizer, feature_to_pixel, coords, layout_audit, col_means

    def fit(self, X, y=None):
        seed_everything(self.random_state)
        if self.torch_num_threads is not None:
            torch.set_num_threads(int(self.torch_num_threads))
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if X.shape[1] > self.image_size * self.image_size:
            # Overlap is allowed, so this is not an error. It is a reminder that
            # many features can share pixels.
            pass

        state, tokenizer, feature_to_pixel, coords, layout_audit, col_means = self._build_training_state(X)
        self.n_features_in_ = int(X.shape[1])
        self.col_means_ = col_means.astype(np.float32)
        self.tokenizer_ = tokenizer
        self.feature_to_pixel_ = feature_to_pixel
        self.feature_coordinates_ = coords
        self.layout_audit_ = layout_audit
        self._state_ = state

        idx = np.arange(X.shape[0])
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=0.25,
            random_state=(0 if self.random_state is None else int(self.random_state) + 100) % (2**32 - 1),
            stratify=None,
        )

        train_ds = _MCQLDataset(state, self.pseudo_mask_rate, tr_idx)
        val_ds = _MCQLDataset(
            state,
            self.val_pseudo_mask_rate,
            va_idx,
            fixed_seed=(0 if self.random_state is None else int(self.random_state) + 300) % (2**32 - 1),
        )

        train_loader = DataLoader(train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=min(self.eval_batch_size, len(val_ds)), shuffle=False, num_workers=0)

        device = get_device(self.device)
        feature_rows = torch.from_numpy(state["feature_rows"]).long().to(device)
        feature_cols = torch.from_numpy(state["feature_cols"]).long().to(device)

        model = GlobalPoolMCQLNet(self.n_bins, self.hidden_channels, self.num_blocks, self.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        best_epoch = 0
        patience_left = int(self.patience)
        history = []

        iterator = range(1, self.max_epochs + 1)
        if self.verbose:
            iterator = tqdm(iterator, desc="MCQL epochs", unit="epoch")

        for epoch in iterator:
            model.train()
            train_sum, train_count = 0.0, 0
            for inp, target, pseudo in train_loader:
                inp = inp.to(device)
                target = target.to(device)
                pseudo = pseudo.to(device)

                opt.zero_grad(set_to_none=True)
                logits = model(inp)
                loss = _mcql_loss(logits, target, pseudo, feature_rows, feature_cols)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

                n = int(pseudo.sum().item())
                train_sum += float(loss.detach().cpu()) * max(n, 1)
                train_count += n

            val = _evaluate_model(model, val_loader, device, feature_rows, feature_cols)
            row = {
                "epoch": int(epoch),
                "train_mcql": train_sum / max(train_count, 1),
                "train_pseudo_count": int(train_count),
                **val,
                "n_params": int(count_parameters(model)),
            }
            history.append(row)

            if val["val_mcql"] < best_val - 1e-5:
                best_val = float(val["val_mcql"])
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
        self.best_epoch_ = best_epoch
        self.best_val_mcql_ = float(best_val)
        return self

    def _build_inference_images(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected X with shape [n, {self.n_features_in_}].")
        M = ~np.isfinite(X)
        X_filled = X.copy()
        rows, cols = np.where(~np.isfinite(X_filled))
        X_filled[rows, cols] = self.col_means_[cols]

        V_features = self.tokenizer_.value_channel(X_filled).astype(np.float32)
        feature_to_flat, occupied_mask_flat, _, _ = build_pixel_groups(self.feature_to_pixel_, self.image_size)
        coord_x, coord_y = coordinate_channels(self.image_size)

        V_base = aggregate_feature_values_to_image(
            V_features=V_features,
            unavailable_features=M,
            feature_to_flat=feature_to_flat,
            occupied_mask_flat=occupied_mask_flat,
            image_size=self.image_size,
        ).astype(np.float32)

        return V_base, coord_x, coord_y, M

    @torch.no_grad()
    def transform(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        V_base, coord_x, coord_y, M = self._build_inference_images(X)
        X_hat = X.copy()
        rows, cols = np.where(~np.isfinite(X_hat))
        X_hat[rows, cols] = self.col_means_[cols]

        if not M.any():
            return X_hat.astype(np.float32)

        device = get_device(self.device)
        self.model_.eval()
        reps = self.tokenizer_.bin_representatives_.astype(np.float32)

        feature_rows = np.asarray([r for r, c in self.feature_to_pixel_], dtype=int)
        feature_cols = np.asarray([c for r, c in self.feature_to_pixel_], dtype=int)

        for start in range(0, X.shape[0], int(self.eval_batch_size)):
            end = min(X.shape[0], start + int(self.eval_batch_size))
            V = torch.from_numpy(V_base[start:end]).float()
            Cx = torch.from_numpy(np.repeat(coord_x[None, ...], end - start, axis=0)).float()
            Cy = torch.from_numpy(np.repeat(coord_y[None, ...], end - start, axis=0)).float()
            inp = torch.stack([V, Cx, Cy], dim=1).to(device)
            logits = self.model_(inp).detach().cpu().numpy()
            z = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(z)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

            M_batch = M[start:end]
            for j, (r, c) in enumerate(zip(feature_rows, feature_cols)):
                rr = M_batch[:, j]
                if not rr.any():
                    continue
                pj = probs[rr, :, int(r), int(c)]
                vals = pj @ reps[j]
                X_hat[start:end, j][rr] = vals.astype(np.float32)

        X_hat[~M] = X[~M]
        return X_hat.astype(np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)
