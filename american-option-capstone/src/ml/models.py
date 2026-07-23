import torch.nn as nn


class PutPricerNet(nn.Module):
    """Two hidden layers, 128 units each. Takes [S0, K, T, r, sigma],
    standardized, and predicts the American put price directly."""

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
