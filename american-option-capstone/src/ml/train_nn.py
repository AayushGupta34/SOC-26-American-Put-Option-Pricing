"""Train the neural-network put pricer.

    python -m ml.train_nn

Loads reports/results/option_data.npz if it already exists (so re-running is
always safe and won't resample), otherwise generates 10,000 random contracts
and labels them with the 500-step binomial price. Saves the trained
checkpoint to reports/results/neural_pricer.pt and the learning curve to
reports/figures/nn_learning_curve.png.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.synthetic_contracts import sample_contracts, label_contracts
from ml.models import PutPricerNet

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "reports", "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")
DATA_PATH = os.path.join(RESULTS_DIR, "option_data.npz")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "neural_pricer.pt")


def load_or_generate_dataset(n=10_000, seed=42):
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH}")
        data = np.load(DATA_PATH)
        return data["X"], data["y"]

    print(f"Generating {n} synthetic contracts...")
    X = sample_contracts(n, seed=seed)
    y = label_contracts(X, steps=500)

    intrinsic = np.maximum(X[:, 1] - X[:, 0], 0.0)
    assert np.isfinite(y).all()
    assert (y >= -1e-10).all()
    assert (y + 1e-8 >= intrinsic).all(), "a label came in below intrinsic value"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(DATA_PATH, X=X, y=y)
    print(f"Saved dataset to {DATA_PATH}")
    return X, y


def train_val_test_split(X, y, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    n_val = int(0.1 * len(X))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


def main():
    torch.manual_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X, y = load_or_generate_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)

    X_train_t = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor((X_val - x_mean) / x_std, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

    model = PutPricerNet()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(300):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        train_loss = float(np.mean(batch_losses))
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0:
            print(f"{epoch:03d}  train={train_loss:.6f}  val={val_loss:.6f}")

    model.load_state_dict(best_state)

    artifact = {
        "model_state": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_order": ["S0", "K", "T", "r", "sigma"],
        "label_steps": 500,
    }
    torch.save(artifact, CHECKPOINT_PATH)
    print(f"\nBest validation MSE: {best_val:.6f}")
    print(f"Saved model to {CHECKPOINT_PATH}")

    plt.figure(figsize=(7, 4))
    plt.plot(history["train"], label="train MSE")
    plt.plot(history["val"], label="validation MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("NN Pricer Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "nn_learning_curve.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
