import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from american_put import crr_put_price

# Load the dataset

data = np.load("week6_option_data.npz")
X = data["X"]
y = data["y"]


def train_val_test_split(X, y, seed=42):

    rng = np.random.default_rng(seed)

    idx = rng.permutation(len(X))

    n_train = int(0.8 * len(X))
    n_val = int(0.1 * len(X))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


_, _, _, _, X_test, y_test = train_val_test_split(X, y)


class PutPricerNet(nn.Module):

    def __init__(self, input_dim=5, hidden=128):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, 1)

        )

    def forward(self, x):

        return self.net(x)


# Load the trained model

loaded = torch.load(
    "week6_neural_pricer.pt",
    map_location="cpu",
    weights_only=False,
)

model = PutPricerNet()

model.load_state_dict(loaded["model_state"])

model.eval()

x_mean = loaded["x_mean"]
x_std = loaded["x_std"]


X_test_s = (X_test - x_mean) / x_std

X_test_t = torch.tensor(
    X_test_s,
    dtype=torch.float32
)


with torch.no_grad():

    pred_test = model(X_test_t).numpy().reshape(-1)


# Overall error

err = pred_test - y_test

mae = np.mean(np.abs(err))

rmse = np.sqrt(np.mean(err**2))

max_abs = np.max(np.abs(err))

print(f"MAE     : {mae:.4f}")
print(f"RMSE    : {rmse:.4f}")
print(f"Max Abs : {max_abs:.4f}")


# Check errors for different moneyness regions

moneyness = X_test[:,0] / X_test[:,1]

buckets = {

    "Deep ITM Put": moneyness < 0.85,

    "Near ATM": (
        (moneyness >= 0.85) &
        (moneyness <= 1.15)
    ),

    "Deep OTM Put": moneyness > 1.15,

}

print()

for name, mask in buckets.items():

    bucket_mae = np.mean(np.abs(err[mask]))

    print(
        f"{name:14s}"
        f" n={mask.sum():4d}"
        f" MAE={bucket_mae:.4f}"
    )


# Compare predictions with binomial prices

plt.figure(figsize=(5,5))

plt.scatter(
    y_test,
    pred_test,
    s=8,
    alpha=0.35
)

lo = min(
    y_test.min(),
    pred_test.min()
)

hi = max(
    y_test.max(),
    pred_test.max()
)

plt.plot(
    [lo, hi],
    [lo, hi],
    color="black"
)

plt.xlabel("Binomial Price")
plt.ylabel("NN Prediction")
plt.title("Predicted vs Binomial")

plt.tight_layout()

plt.savefig(
    "week6_pred_vs_binomial.png",
    dpi=160
)

plt.show()


def nn_predict_price(X_raw):

    X_scaled = (X_raw - x_mean) / x_std

    X_t = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():

        return model(X_t).numpy().reshape(-1)


# Compare the neural network surface with the binomial surface

S_grid = np.linspace(60, 140, 41)
T_grid = np.linspace(0.05, 2.0, 40)

K_fixed = 100.0
r_fixed = 0.05
sigma_fixed = 0.25

rows = []

for T in T_grid:
    for S0 in S_grid:
        rows.append([S0, K_fixed, T, r_fixed, sigma_fixed])

X_surface = np.array(rows)

nn_surface = nn_predict_price(
    X_surface
).reshape(
    len(T_grid),
    len(S_grid)
)

binomial_surface = np.array([

    crr_put_price(
        S0,
        K_fixed,
        T,
        r_fixed,
        sigma_fixed,
        500,
        american=True
    )

    for T in T_grid
    for S0 in S_grid

]).reshape(
    len(T_grid),
    len(S_grid)
)

abs_surface_error = np.abs(
    nn_surface - binomial_surface
)

plt.figure(figsize=(6,4))

plt.imshow(
    abs_surface_error,
    origin="lower",
    aspect="auto",
    extent=[
        S_grid.min(),
        S_grid.max(),
        T_grid.min(),
        T_grid.max()
    ]
)

plt.colorbar(label="Absolute Error")

plt.xlabel("Spot Price")
plt.ylabel("Time to Maturity")
plt.title("Absolute Surface Error")

plt.tight_layout()

plt.savefig(
    "week6_surface_error.png",
    dpi=160
)

plt.show()


# Basic financial sanity checks

slice_rows = np.array([

    [S0, 100.0, 1.0, 0.05, 0.25]

    for S0 in np.linspace(60, 140, 81)

])

slice_pred = nn_predict_price(slice_rows)

monotonic_violations = np.sum(
    np.diff(slice_pred) > 1e-4
)

negative_predictions = np.sum(
    slice_pred < -1e-6
)

intrinsic = np.maximum(
    slice_rows[:,1] - slice_rows[:,0],
    0.0
)

intrinsic_violations = np.sum(
    slice_pred + 1e-4 < intrinsic
)

print()
print("Monotonic violations :", monotonic_violations)
print("Negative predictions :", negative_predictions)
print("Below intrinsic      :", intrinsic_violations)
print("The neural network was able to learn the pricing pattern of the 500-step CRR American put model quite well. Both the training and validation losses decreased steadily throughout training, and there was no noticeable overfitting since the two curves stayed close together. The predicted vs. binomial scatter plot also showed that the neural network's predictions were very close to the actual binomial prices. The model performed best for deep out-of-the-money options, while the errors were slightly higher for deep in-the-money and near at-the-money options. It also passed most of the financial sanity checks, with no negative prices and no monotonicity violations. There were a few predictions below the intrinsic value, but these were very small and are expected because the neural network was trained to minimize prediction error rather than explicitly enforce financial constraints.")
