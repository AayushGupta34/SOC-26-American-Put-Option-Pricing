"""
Generates the figures that don't already exist elsewhere in the repo:

    reports/figures/put_payoff_diagram.png       - basic long-put payoff intuition
    reports/figures/exercise_boundary_standalone.png - binomial exercise boundary,
                                                        as its own figure (previously
                                                        only existed folded into
                                                        week4_AayushGupta/week4_report.py)
    reports/figures/nn_vs_binomial_full_grid.png - predicted-vs-benchmark scatter,
                                                    regenerated against the full
                                                    Week 9 shared grid rather than
                                                    just week6's original test split

Already-existing figures (week8_AayushGupta/exercise_region.png,
week6/week6_pred_vs_binomial.png, week6/week6_surface_error.png) are reused
as-is and not regenerated here.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_utils import compute_boundary_by_step
from option_contract import OptionContract
from rl_policy import WEEK8_CONTRACT

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def plot_put_payoff_diagram(K=100.0, premium=7.95, path=None):
    S_T = np.linspace(0, 2 * K, 200)
    payoff = np.maximum(K - S_T, 0.0)
    profit = payoff - premium

    plt.figure(figsize=(6, 4.5))
    plt.plot(S_T, payoff, label="Payoff at expiry: max(K - S_T, 0)", lw=2)
    plt.plot(S_T, profit, label=f"Profit (payoff - premium of {premium:.2f})", lw=2, ls="--")
    plt.axhline(0, color="black", lw=0.8)
    plt.axvline(K, color="gray", ls=":", lw=1, label=f"Strike K = {K:.0f}")
    plt.xlabel("Stock price at expiry, $S_T$")
    plt.ylabel("Value")
    plt.title("Long American Put: Payoff and Profit Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_exercise_boundary_standalone(contract: OptionContract, path=None):
    boundary_by_step = compute_boundary_by_step(contract)
    dt = contract.T / contract.steps
    steps_sorted = sorted(boundary_by_step)
    times = [i * dt for i in steps_sorted]
    spots = [boundary_by_step[i] for i in steps_sorted]

    plt.figure(figsize=(7, 4))
    plt.plot(times, spots, marker=".")
    plt.axhline(contract.K, color="gray", linestyle="--", linewidth=1, label="Strike K")
    plt.xlabel("Time")
    plt.ylabel("Highest exercise stock price")
    plt.title(
        f"American Put Exercise Boundary (S0={contract.S0}, K={contract.K}, "
        f"T={contract.T}, r={contract.r}, sigma={contract.sigma})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_nn_vs_binomial_full_grid(df: pd.DataFrame, path=None):
    x, y = df["binomial_price"], df["nn_price"]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5, s=14)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
    plt.xlabel("Binomial benchmark price")
    plt.ylabel("NN predicted price")
    plt.title(f"NN price vs binomial benchmark (full shared grid, n={len(df)})")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    grid_csv = os.path.join(OUTPUT_DIR, "pricing_comparison_grid.csv")
    if not os.path.exists(grid_csv):
        raise FileNotFoundError(f"{grid_csv} not found -- run compare_all.py first.")
    df = pd.read_csv(grid_csv)

    week8_price_path = os.path.join(OUTPUT_DIR, "summary.txt")
    premium = 7.95  # falls back to the printed week8-contract binomial price from compare_all.py
    if os.path.exists(week8_price_path):
        with open(week8_price_path) as f:
            for line in f:
                if "binomial (American) price" in line:
                    premium = float(line.strip().split(":")[-1])

    plot_put_payoff_diagram(
        K=WEEK8_CONTRACT.K, premium=premium,
        path=os.path.join(FIGURES_DIR, "put_payoff_diagram.png"),
    )
    print("saved put_payoff_diagram.png")

    plot_exercise_boundary_standalone(
        WEEK8_CONTRACT,
        path=os.path.join(FIGURES_DIR, "exercise_boundary_standalone.png"),
    )
    print("saved exercise_boundary_standalone.png")

    plot_nn_vs_binomial_full_grid(
        df, path=os.path.join(FIGURES_DIR, "nn_vs_binomial_full_grid.png"),
    )
    print("saved nn_vs_binomial_full_grid.png")


if __name__ == "__main__":
    main()
