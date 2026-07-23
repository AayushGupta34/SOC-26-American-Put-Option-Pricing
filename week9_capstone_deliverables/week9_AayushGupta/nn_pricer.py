"""
Loads the Week 6 neural-network pricer checkpoint for reuse in the Week 9
comparison, without re-triggering Week 6's training scripts on import
(week6/PartA.py, PartB.py, and PartC.py all execute top-level code as soon as
they're imported, since they were written as standalone scripts).

IMPORTANT: `PutPricerNet` here is a literal copy of the class defined in
week6/PartB.py (and re-declared identically in week6/PartC.py). torch.save
only stores weights (state_dict()), not the class definition, so this
architecture must stay byte-for-byte in sync with week6/PartB.py's version.
If you ever change the Week 6 architecture, update this copy too, or a
future evaluator loading the same .pt file here will get shape-mismatch
errors that have nothing to do with the actual pricing question.
"""

import os

import numpy as np
import torch
from torch import nn

from option_contract import OptionContract

DEFAULT_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "week6", "week6_neural_pricer.pt"
)


class PutPricerNet(nn.Module):
    def __init__(self, input_dim=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class NNPricer:
    """Thin wrapper so the comparison code can call `.predict(contract)`."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find {checkpoint_path}. Run week6/PartA.py then "
                f"week6/PartB.py first to generate week6_neural_pricer.pt "
                f"(see the Week 6 README's 'Note on week6_neural_pricer.pt')."
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
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_t).numpy().reshape(-1)

    def predict(self, contract: OptionContract) -> float:
        row = np.array([contract.as_row()], dtype=np.float64)
        return float(self.predict_rows(row)[0])

    def predict_many(self, contracts) -> np.ndarray:
        rows = np.array([c.as_row() for c in contracts], dtype=np.float64)
        return self.predict_rows(rows)
