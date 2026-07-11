import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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


X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)


x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)

x_std = np.where(x_std == 0, 1.0, x_std)

X_train_s = (X_train - x_mean) / x_std
X_val_s = (X_val - x_mean) / x_std
X_test_s = (X_test - x_mean) / x_std



torch.manual_seed(42)

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)

X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)


train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=256,
    shuffle=True
)




class PutPricerNet(nn.Module):

    def __init__(self, input_dim=5, hidden=128):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden,1)

        )

    def forward(self,x):
        return self.net(x)


model = PutPricerNet()



loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

history = {
    "train":[],
    "val":[]
}

best_val = float("inf")
best_state = None


for epoch in range(300):

    model.train()

    batch_losses=[]

    for xb,yb in train_loader:

        pred = model(xb)

        loss = loss_fn(pred,yb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_losses.append(loss.item())

    model.eval()

    with torch.no_grad():

        val_pred = model(X_val_t)

        val_loss = loss_fn(val_pred,y_val_t).item()

    train_loss = float(np.mean(batch_losses))

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    if val_loss < best_val:

        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 25 == 0:

        print(
            f"{epoch:03d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )


model.load_state_dict(best_state)

artifact = {
    "model_state": model.state_dict(),
    "x_mean": x_mean,
    "x_std": x_std,
    "feature_order": [
        "S0",
        "K",
        "T",
        "r",
        "sigma"
    ],
    "label_steps":500
}

torch.save(
    artifact,
    "week6_neural_pricer.pt"
)

print("\nBest validation MSE:", best_val)
print("Saved model to week6_neural_pricer.pt")



plt.figure(figsize=(7,4))

plt.plot(history["train"],label="train MSE")
plt.plot(history["val"],label="validation MSE")

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title("Week 6 Neural Pricer Learning Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    "week6_learning_curve.png",
    dpi=160
)

plt.show()
