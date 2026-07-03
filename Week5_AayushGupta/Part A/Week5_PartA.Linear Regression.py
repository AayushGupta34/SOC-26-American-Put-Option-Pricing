import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# generating x values in the required range
x = rng.uniform(0, 10, size=200)

# extra is noise
y = 2.5 * x + 1.0 + rng.normal(0, 0.5, size=200)

# shuffling indices before splitting
idx = rng.permutation(len(x))

# 80-20 split
train_end = int(0.8 * len(x))

x_train = x[idx[:train_end]]
y_train = y[idx[:train_end]]

x_val = x[idx[train_end:]]
y_val = y[idx[train_end:]]

w = 0.0
b = 0.0

#learning rate given in video tut
lr = 0.01

#no of iteration
epochs = 500

train_losses = []

for _ in range(epochs):

    # Prediction
    y_hat = w * x_train + b

    # Error
    error = y_hat - y_train

    # Gradients
    grad_w = (2 / len(x_train)) * np.dot(error, x_train)
    grad_b = (2 / len(x_train)) * np.sum(error)

    # Update
    w -= lr * grad_w
    b -= lr * grad_b

    # Track loss
    loss = np.mean(error ** 2)
    train_losses.append(loss)

y_val_pred = w * x_val + b

val_loss = np.mean((y_val_pred - y_val) ** 2)

print("Learned w =", w, "| True w = 2.5")
print("Learned b =", b, "| True b = 1.0")

print("Train MSE =", train_losses[-1])
print("Validation MSE =", val_loss)

if abs(train_losses[-1] - val_loss) < 0.1:
    print("Train and validation errors are close, model generalizes.")
else:
    print("Large difference between train and validation error.")

plt.scatter(x_train, y_train, label="Training Data")

x_line = np.linspace(0,10,100)

y_line = w * x_line + b

plt.plot(x_line, y_line, label="Fitted Line")

plt.legend()
plt.savefig("plots/fitted_line.png", dpi=300, bbox_inches="tight")
plt.show()
