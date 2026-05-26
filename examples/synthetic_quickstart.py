import numpy as np

from deepinpainting import DeepInpaintingImputer


def main():
    rng = np.random.default_rng(0)

    n_samples, n_features, latent_dim = 24, 40, 3
    latent = rng.normal(size=(n_samples, latent_dim))
    loadings = rng.normal(size=(latent_dim, n_features))
    X = latent @ loadings + 0.1 * rng.normal(size=(n_samples, n_features))

    mask = rng.random(X.shape) < 0.10
    X_missing = X.copy()
    X_missing[mask] = np.nan

    imputer = DeepInpaintingImputer(
        n_bins=8,
        image_size=16,
        reducer="pca",
        hidden_channels=8,
        num_blocks=1,
        max_epochs=1,
        patience=1,
        batch_size=6,
        eval_batch_size=8,
        random_state=0,
        torch_num_threads=1,
        device="cpu",
        verbose=False,
    )

    X_completed = imputer.fit_transform(X_missing)
    rmse = np.sqrt(np.mean((X[mask] - X_completed[mask]) ** 2))

    print(f"completed shape: {X_completed.shape}")
    print(f"masked RMSE: {rmse:.4f}")
    print("layout audit:", imputer.layout_audit_)


if __name__ == "__main__":
    main()
