import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# using more than 500 points
x = np.linspace(0, 2*np.pi, 600)

# noisy sine curve
y = np.sin(x) + rng.normal(0, 0.1, size=600)

idx = rng.permutation(len(x))

train_end = int(0.8 * len(x))

x_train = x[idx[:train_end]]
y_train = y[idx[:train_end]]

x_val = x[idx[train_end:]]
y_val = y[idx[train_end:]]

# 1 input, 16 hidden neurons
d_in = 1
h_dim = 16

W1 = 0.1 * rng.standard_normal((h_dim, d_in))
b1 = np.zeros(h_dim)

W2 = 0.1 * rng.standard_normal((1, h_dim))
b2 = np.zeros(1)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward(x, W1, b1, W2, b2):

    # hidden layer
    z1 = W1 @ x + b1.reshape(-1,1)

    h = relu(z1)

    # output layer
    y_hat = W2 @ h + b2

    return y_hat, h, z1

x_train = x_train.reshape(1,-1)
y_train = y_train.reshape(1,-1)

x_val = x_val.reshape(1,-1)
y_val = y_val.reshape(1,-1)

lr = 0.01
epochs = 1000

train_losses = []
val_losses = []

for _ in range(epochs):

    #forward pass
    y_hat, h, z1 = forward(x_train, W1, b1, W2, b2)

    #training error
    error = y_hat - y_train

    #mse loss
    loss = np.mean(error**2)

    train_losses.append(loss)

    n = x_train.shape[1]

    #derivative wrt output
    d_yhat = (2/n) * error

    dW2 = d_yhat @ h.T
    db2 = np.sum(d_yhat)

    # backprop to hidden layer
    dh = W2.T @ d_yhat

    # relu derivative
    dz1 = dh * relu_derivative(z1)

    # gradients first layer
    dW1 = dz1 @ x_train.T
    db1 = np.sum(dz1, axis=1)

    # parameter update
    W1 -= lr * dW1
    b1 -= lr * db1

    W2 -= lr * dW2
    b2 -= lr * db2

    # validation loss
    val_pred, _, _ = forward(x_val, W1, b1, W2, b2)

    val_loss = np.mean((val_pred - y_val)**2)

    val_losses.append(val_loss)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")

plt.legend()
plt.savefig("plots/train_val_loss.png", dpi=300, bbox_inches="tight")
plt.show()

x_dense = np.linspace(0, 2*np.pi, 1000)

x_dense = x_dense.reshape(1,-1)

pred, _, _ = forward(x_dense, W1, b1, W2, b2)

plt.plot(x_dense.flatten(),
         np.sin(x_dense.flatten()),
         label="True Curve")

plt.plot(x_dense.flatten(),
         pred.flatten(),
         label="Model Prediction")

plt.legend()
plt.title("True Curve vs Prediction")
plt.savefig("plots/prediction_curve.png", dpi=300, bbox_inches="tight")
plt.show()

val_pred, _, _ = forward(x_val, W1, b1, W2, b2)

# average absolute error
mae = np.mean(np.abs(val_pred - y_val))

# worst error
max_error = np.max(np.abs(val_pred - y_val))

print("\nFinal Results")

print("Validation MAE =", mae)

print("Max Absolute Error =", max_error)

print("Final Train Loss =", train_losses[-1])

print("Final Validation Loss =", val_losses[-1])
