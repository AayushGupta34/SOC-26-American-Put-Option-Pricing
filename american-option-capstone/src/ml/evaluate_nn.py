"""Load the trained NN pricer and evaluate it on its own held-out test split.

    python -m ml.evaluate_nn

`NNPricer` is what the rest of the project (evaluation/comparison.py) imports
to get predictions -- it just loads the checkpoint, it doesn't retrain
anything, so importing it is cheap.
"""

import os

import numpy as np
import torch

from ml.models import PutPricerNet
from ml.train_nn import DATA_PATH, CHECKPOINT_PATH, train_val_test_split
from data.synthetic_contracts import OptionContract


class NNPricer:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"{checkpoint_path} not found. Run `python -m ml.train_nn` first."
            )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = PutPricerNet()
        self.model.load_state_dict(loaded["model_state"])
        self.model.eval()
        self.x_mean = loaded["x_mean"]
        self.x_std = loaded["x_std"]
        self.feature_order = loaded.get("feature_order", ["S0", "K", "T", "r", "sigma"])

    def predict_rows(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled = (X_raw - self.x_mean) / self.x_std
        with torch.no_grad():
            return self.model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().reshape(-1)

    def predict(self, contract: OptionContract) -> float:
        return float(self.predict_rows(np.array([contract.as_row()], dtype=np.float64))[0])

    def predict_many(self, contracts) -> np.ndarray:
        return self.predict_rows(np.array([c.as_row() for c in contracts], dtype=np.float64))


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python -m ml.train_nn` first.")

    data = np.load(DATA_PATH)
    _, _, _, _, X_test, y_test = train_val_test_split(data["X"], data["y"])

    pricer = NNPricer()
    pred_test = pricer.predict_rows(X_test)

    err = pred_test - y_test
    print(f"MAE     : {np.mean(np.abs(err)):.4f}")
    print(f"RMSE    : {np.sqrt(np.mean(err ** 2)):.4f}")
    print(f"Max abs : {np.max(np.abs(err)):.4f}")

    moneyness = X_test[:, 0] / X_test[:, 1]
    buckets = {
        "Deep ITM put": moneyness < 0.85,
        "Near ATM": (moneyness >= 0.85) & (moneyness <= 1.15),
        "Deep OTM put": moneyness > 1.15,
    }
    print()
    for name, mask in buckets.items():
        print(f"{name:14s} n={mask.sum():4d}  MAE={np.mean(np.abs(err[mask])):.4f}")


if __name__ == "__main__":
    main()
