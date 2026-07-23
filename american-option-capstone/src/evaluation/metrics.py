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


def count_intrinsic_violations(df, price_col: str = "nn_price", tol: float = 1e-8) -> int:
    """How many rows priced the put below its intrinsic value max(K-S0, 0)."""
    intrinsic = np.maximum(df["K"] - df["S0"], 0.0)
    return int((df[price_col] + tol < intrinsic).sum())


def put_spot_monotonicity_check(predict_fn, K=100, T=1.0, r=0.05, sigma=0.25, n=41):
    """A put's price should never increase as spot rises. Returns the
    violating (spot_lo, spot_hi, price_lo, price_hi) pairs, if any."""
    spots = np.linspace(60, 140, n)
    prices = [predict_fn(S0=S, K=K, T=T, r=r, sigma=sigma) for S in spots]
    violations = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1] + 1e-6:
            violations.append((spots[i - 1], spots[i], prices[i - 1], prices[i]))
    return violations
