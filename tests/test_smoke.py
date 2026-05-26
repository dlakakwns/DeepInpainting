import numpy as np

from deepinpainting import DeepInpaintingImputer


def test_deepinpainting_smoke():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(12, 20)).astype("float32")
    mask = rng.random(X.shape) < 0.1
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
        batch_size=4,
        eval_batch_size=6,
        random_state=1,
        torch_num_threads=1,
        device="cpu",
    )
    out = imputer.fit_transform(X_missing)

    assert out.shape == X.shape
    assert np.isfinite(out).all()
    assert np.allclose(out[~mask], X[~mask])
