import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing.binomial import exercise_boundary_by_step

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")
RESULTS_DIR = os.path.join(REPO_ROOT, "reports", "results")


def plot_put_payoff_diagram(K, premium, path):
    S_T = np.linspace(0, 2 * K, 200)
    payoff = np.maximum(K - S_T, 0.0)
    profit = payoff - premium

    plt.figure(figsize=(6, 4.5))
    plt.plot(S_T, payoff, label="Payoff at expiry: max(K - S_T, 0)", lw=2)
    plt.plot(S_T, profit, label=f"Profit (payoff - premium of {premium:.2f})", lw=2, ls="--")
    plt.axhline(0, color="black", lw=0.8)
    plt.axvline(K, color="gray", ls=":", lw=1, label=f"Strike K = {K:.0f}")
    plt.xlabel("Stock price at expiry, S_T")
    plt.ylabel("Value")
    plt.title("Long American Put: Payoff and Profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_exercise_boundary(S0, K, T, r, sigma, steps, path):
    boundary = exercise_boundary_by_step(S0, K, T, r, sigma, steps, option_type="put")
    dt = T / steps
    steps_sorted = sorted(boundary)
    times = [i * dt for i in steps_sorted]
    spots = [boundary[i] for i in steps_sorted]

    plt.figure(figsize=(7, 4))
    plt.plot(times, spots, marker=".")
    plt.axhline(K, color="gray", linestyle="--", linewidth=1, label="Strike K")
    plt.xlabel("Time")
    plt.ylabel("Highest exercise stock price")
    plt.title(f"American Put Exercise Boundary (S0={S0}, K={K}, T={T}, r={r}, sigma={sigma})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_nn_vs_binomial(df: pd.DataFrame, path):
    x, y = df["binomial_price"], df["nn_price"]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, s=14)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
    plt.xlabel("Binomial benchmark price")
    plt.ylabel("NN predicted price")
    plt.title(f"NN price vs binomial benchmark (n={len(df)})")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    grid_csv = os.path.join(RESULTS_DIR, "pricing_comparison.csv")
    if not os.path.exists(grid_csv):
        raise FileNotFoundError(f"{grid_csv} not found -- run `python -m evaluation.comparison` first.")
    df = pd.read_csv(grid_csv)

    plot_put_payoff_diagram(K=100.0, premium=7.95, path=os.path.join(FIGURES_DIR, "put_payoff_diagram.png"))
    plot_exercise_boundary(100.0, 100.0, 1.0, 0.05, 0.25, 50,
                           path=os.path.join(FIGURES_DIR, "exercise_boundary.png"))
    plot_nn_vs_binomial(df, path=os.path.join(FIGURES_DIR, "nn_vs_binomial.png"))
    print(f"figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
