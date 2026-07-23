"""
Regression metrics + financial sanity checks for the NN pricer vs. the
binomial benchmark. week6/PartC.py already computes MAE/RMSE/moneyness-bucket
error on its own test split; these functions do the same thing but operate on
whatever contract grid/DataFrame is handed to them, so Week 9 can reuse them
across the full shared grid instead of just Week 6's original test split.
"""

import numpy as np


def pricing_metrics(y_true, y_pred, eps: float = 1e-8) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_bias": float(np.mean(err)),
        "median_abs_error": float(np.median(np.abs(err))),
        "mean_relative_error": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps))),
    }


def intrinsic_put_value(S: float, K: float) -> float:
    return max(K - S, 0.0)


def count_intrinsic_violations(df, price_col: str = "nn_price", tol: float = 1e-8) -> int:
    """Rows where the NN priced the put below its intrinsic value."""
    intrinsic = np.maximum(df["K"] - df["S0"], 0.0)
    violations = df[price_col] + tol < intrinsic
    return int(violations.sum())


def put_spot_monotonicity_check(predict_fn, K=100, T=1.0, r=0.05, sigma=0.25, n=41):
    """A put's price should be non-increasing in spot. Returns any violating pairs."""
    spots = np.linspace(60, 140, n)
    prices = [predict_fn(S0=S, K=K, T=T, r=r, sigma=sigma) for S in spots]
    increases = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1] + 1e-6:
            increases.append((spots[i - 1], spots[i], prices[i - 1], prices[i]))
    return increases
