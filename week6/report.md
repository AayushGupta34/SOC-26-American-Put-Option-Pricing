# Week 6 Report — Neural Network Approximation of an American Put Pricer

## 1. Dataset (Part A)

10,000 synthetic American put contracts sampled independently and
uniformly (`seed=42`):

| Feature | Range |
|---|---|
| S0 (spot) | 60 – 140 |
| K (strike) | 80 – 120 |
| T (maturity, yrs) | 0.05 – 2.0 |
| r (risk-free rate) | 0.00 – 0.10 |
| sigma (volatility) | 0.10 – 0.50 |

Each contract labeled with the Week 4 CRR binomial pricer at a fixed
500 steps, `american=True`.

**Sanity checks — actual output from running `PartA.py`:**

```
Label range        : 1.66e-51  to  59.68
Intrinsic violations: 0   (out of 10,000)
```

- Finite (`np.isfinite(y).all()`): passed.
- Non-negative (`y >= -1e-10`): passed.
- At least intrinsic value (`y + 1e-8 >= intrinsic`): passed for all 10,000 rows.

Saved to `week6_option_data.npz`.

## 2. Training (Part B)

- Split: 80% train / 10% val / 10% test (8,000 / 1,000 / 1,000).
- Standardization: `(X - mean) / std` using training-set statistics only.
- Architecture: `Linear(5→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1)`, Adam (lr=1e-3), MSE loss, batch size 256, 300 epochs.
- Checkpointing: state dict with the lowest validation MSE across all 300 epochs is saved to `week6_neural_pricer.pt`, together with `x_mean`, `x_std`, `feature_order`, and `label_steps` (see README for exact lines).

Learning curve: `week6_learning_curve.png`.

**Best validation MSE:** `[fill in — printed at the end of your PartB.py run: "Best validation MSE: <value>"]`

## 3. Evaluation (Part C)

**Overall test-set error (n = 1,000):**

| Metric | Value |
|---|---|
| MAE | `[fill in from console: "MAE     : ..."]` |
| RMSE | `[fill in from console: "RMSE    : ..."]` |
| Max absolute error | `[fill in from console: "Max Abs : ..."]` |

**Error by moneyness bucket (moneyness = S0/K):**

| Bucket | Definition | n | MAE |
|---|---|---|---|
| Deep ITM Put | moneyness < 0.85 | `[n]` | `[fill in]` |
| Near ATM | 0.85 ≤ moneyness ≤ 1.15 | `[n]` | `[fill in]` |
| Deep OTM Put | moneyness > 1.15 | `[n]` | `[fill in]` |

Plots: `week6_pred_vs_binomial.png` (predicted vs. binomial scatter), `week6_surface_error.png` (absolute error surface, K=100, r=5%, sigma=25%).

**Sanity checks (81-point S0 slice, K=100, T=1, r=5%, sigma=25%):**

| Check | Result |
|---|---|
| Monotonic violations | `[fill in]` |
| Negative predictions | `[fill in]` |
| Predictions below intrinsic value | `[fill in]` |

*(These six values were printed to your console when you ran `PartB.py`/`PartC.py` locally but weren't included in what was shared, so they're left blank here rather than invented.)*

## 4. Reflection

As printed by `PartC.py`:

> The neural network was able to learn the pricing pattern of the 500-step CRR American put model quite well. Both the training and validation losses decreased steadily throughout training, and there was no noticeable overfitting since the two curves stayed close together. The predicted vs. binomial scatter plot also showed that the neural network's predictions were very close to the actual binomial prices. The model performed best for deep out-of-the-money options, while the errors were slightly higher for deep in-the-money and near at-the-money options. It also passed most of the financial sanity checks, with no negative prices and no monotonicity violations. There were a few predictions below the intrinsic value, but these were very small and are expected because the neural network was trained to minimize prediction error rather than explicitly enforce financial constraints.

## 5. Files included in this submission

- `american_put.py`, `PartA.py`, `PartB.py`, `PartC.py` — original pipeline code, unedited.
- `week6_learning_curve.png`, `week6_pred_vs_binomial.png`, `week6_surface_error.png` — required plots.
- `week6_neural_pricer.pt` — add from your own `PartB.py` run (see README).
- `README.md`, `report.md`.
