import numpy as np

from mcql_deepinsight import MCQLImputer


def main():
    rng = np.random.default_rng(0)

    # Synthetic high-dimensional tabular data with latent structure.
    n, p, latent = 40, 120, 4
    Z = rng.normal(size=(n, latent))
    W = rng.normal(size=(latent, p))
    X = Z @ W + 0.1 * rng.normal(size=(n, p))

    mask = rng.random(X.shape) < 0.10
    X_missing = X.copy()
    X_missing[mask] = np.nan

    imputer = MCQLImputer(
        n_bins=16,
        image_size=32,
        reducer="pca",
        hidden_channels=16,
        num_blocks=2,
        max_epochs=5,
        patience=2,
        batch_size=8,
        eval_batch_size=16,
        random_state=0,
        device="cpu",
        verbose=True,
    )

    X_completed = imputer.fit_transform(X_missing)
    rmse = np.sqrt(np.mean((X[mask] - X_completed[mask]) ** 2))

    print(f"Completed shape: {X_completed.shape}")
    print(f"Masked RMSE: {rmse:.4f}")
    print("Layout audit:", imputer.layout_audit_)
    print("Best validation MCQL:", imputer.best_val_mcql_)


if __name__ == "__main__":
    main()
