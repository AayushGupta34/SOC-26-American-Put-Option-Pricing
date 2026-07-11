import os
import numpy as np
from american_put import crr_put_price


def sample_contracts(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    S0 = rng.uniform(60.0, 140.0, size=n)
    K = rng.uniform(80.0, 120.0, size=n)
    T = rng.uniform(0.05, 2.0, size=n)
    r = rng.uniform(0.00, 0.10, size=n)
    sigma = rng.uniform(0.10, 0.50, size=n)

    return np.column_stack([S0, K, T, r, sigma])


def generate_labels(X: np.ndarray, steps: int = 500) -> np.ndarray:

    y = np.empty(len(X), dtype=np.float64)

    for i, (S0, K, T, r, sigma) in enumerate(X):

        y[i] = crr_put_price(
            S0=float(S0),
            K=float(K),
            T=float(T),
            r=float(r),
            sigma=float(sigma),
            steps=steps,
            american=True,
        )

    return y

# Generate OR Load Dataset

if os.path.exists("week6_option_data.npz"):

    print("Loading existing dataset...")

    data = np.load("week6_option_data.npz")

    X = data["X"]
    y = data["y"]

else:

    print("Generating dataset...")

    X = sample_contracts(10000)

    y = generate_labels(X, steps=500)

    # Sanity checks

    intrinsic = np.maximum(X[:, 1] - X[:, 0], 0.0)

    assert np.isfinite(y).all()
    assert (y >= -1e-10).all()
    assert (y + 1e-8 >= intrinsic).all()

    print("Label range:", y.min(), y.max())
    print("Intrinsic violations:", np.sum(y + 1e-8 < intrinsic))

    np.savez_compressed(
        "week6_option_data.npz",
        X=X,
        y=y
    )

    print("Dataset saved to week6_option_data.npz")


print("Dataset shape:", X.shape)
print("Labels shape :", y.shape)
