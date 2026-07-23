"""Run the full binomial vs. NN vs. RL comparison.

    python -m evaluation.comparison

Writes CSVs and a plain-text summary to reports/results/, and generates the
figures in reports/figures/.
"""

import os

import numpy as np
import pandas as pd

from data.synthetic_contracts import make_shared_grid, OptionContract
from pricing.binomial import crr_price
from ml.evaluate_nn import NNPricer
from evaluation.metrics import pricing_metrics, count_intrinsic_violations, put_spot_monotonicity_check
from evaluation.plots import plot_put_payoff_diagram, plot_exercise_boundary, plot_nn_vs_binomial
from rl.env import TRAINING_CONTRACT
from rl.evaluate_policy import build_policy_comparison, boundary_agreement, evaluate_dqn_across_grid, load_trained_dqn, make_dqn_policy

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "reports", "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")


def add_slicing_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratio = df["S0"] / df["K"]
    df["moneyness_bucket"] = pd.cut(ratio, bins=[0.0, 0.9, 1.1, 10.0], labels=["ITM put", "ATM", "OTM put"])
    df["maturity_bucket"] = np.where(df["T"] <= 0.5, "short (<=0.5y)", "long (>0.5y)")
    df["vol_bucket"] = np.where(df["sigma"] <= 0.20, "low vol (<=0.20)", "high vol (>0.20)")
    return df


def summarize_by(df: pd.DataFrame, by_col: str) -> pd.DataFrame:
    grouped = df.groupby(by_col, observed=True)
    return grouped.agg(
        nn_mae=("nn_error", lambda x: float(np.mean(np.abs(x)))),
        nn_rmse=("nn_error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
        nn_bias=("nn_error", "mean"),
        binomial_mean=("binomial_price", "mean"),
        nn_mean=("nn_price", "mean"),
        count=(by_col, "size"),
    ).reset_index()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Building the shared contract grid...")
    contracts = make_shared_grid()
    print(f"  {len(contracts)} contracts")

    print("Pricing every contract with the binomial tree and the NN...")
    nn_pricer = NNPricer()
    rows = []
    for c in contracts:
        binomial_price = crr_price(c.S0, c.K, c.T, c.r, c.sigma, c.steps, option_type="put", american=True)
        nn_price = nn_pricer.predict(c)
        rows.append({
            "S0": c.S0, "K": c.K, "T": c.T, "r": c.r, "sigma": c.sigma, "steps": c.steps,
            "binomial_price": binomial_price, "nn_price": nn_price,
            "nn_error": nn_price - binomial_price,
        })
    df = add_slicing_columns(pd.DataFrame(rows))
    df.to_csv(os.path.join(RESULTS_DIR, "pricing_comparison.csv"), index=False)

    overall = pricing_metrics(df["binomial_price"], df["nn_price"])
    intrinsic_violations = count_intrinsic_violations(df)
    print(f"\nNN vs binomial, n={len(df)}:")
    for k, v in overall.items():
        print(f"  {k:22s}: {v:.6f}")
    print(f"  {'intrinsic_violations':22s}: {intrinsic_violations} / {len(df)}")

    for by_col, fname in [("moneyness_bucket", "bucket_moneyness.csv"),
                           ("maturity_bucket", "bucket_maturity.csv"),
                           ("vol_bucket", "bucket_volatility.csv")]:
        summary = summarize_by(df, by_col)
        summary.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
        print(f"\n-- by {by_col} --\n{summary.to_string(index=False)}")

    increases = put_spot_monotonicity_check(
        lambda S0, K, T, r, sigma: nn_pricer.predict(
            OptionContract(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=100)
        )
    )
    print(f"\nspot-monotonicity violations: {len(increases)}")

    print(f"\nRL: DQN vs. baselines on its own training contract {TRAINING_CONTRACT}")
    rl_comparison = build_policy_comparison(episodes=10_000)
    rl_comparison.to_csv(os.path.join(RESULTS_DIR, "rl_policy_comparison.csv"), index=False)
    print(rl_comparison.to_string(index=False))

    training_binomial_price = crr_price(**TRAINING_CONTRACT, option_type="put", american=True)
    online = load_trained_dqn()
    agreement = boundary_agreement(make_dqn_policy(online))
    print(f"binomial (American) price for this contract: {training_binomial_price:.4f}")
    print(f"boundary agreement: {agreement:.4f}")

    print(f"\nRL: DQN run across the full {len(contracts)}-contract grid (out-of-distribution check)...")
    rl_grid_df = add_slicing_columns(evaluate_dqn_across_grid(contracts, episodes=300))
    rl_grid_df.to_csv(os.path.join(RESULTS_DIR, "rl_policy_grid_comparison.csv"), index=False)

    rl_grid_summary = {
        "mean_abs_dqn_gap_vs_binomial": float(rl_grid_df["dqn_gap_vs_binomial"].abs().mean()),
        "fraction_where_dqn_beats_hold": float((rl_grid_df["dqn_value"] > rl_grid_df["hold_value"]).mean()),
    }
    for k, v in rl_grid_summary.items():
        print(f"  {k}: {v:.4f}")

    rl_grid_bucket = rl_grid_df.groupby("moneyness_bucket", observed=True).agg(
        mean_dqn_value=("dqn_value", "mean"),
        mean_hold_value=("hold_value", "mean"),
        mean_binomial_price=("binomial_price", "mean"),
        mean_abs_gap=("dqn_gap_vs_binomial", lambda x: float(np.mean(np.abs(x)))),
        count=("moneyness_bucket", "size"),
    ).reset_index()
    rl_grid_bucket.to_csv(os.path.join(RESULTS_DIR, "rl_grid_bucket_moneyness.csv"), index=False)
    print(rl_grid_bucket.to_string(index=False))

    print("\nGenerating figures...")
    plot_put_payoff_diagram(K=TRAINING_CONTRACT["K"], premium=training_binomial_price,
                             path=os.path.join(FIGURES_DIR, "put_payoff_diagram.png"))
    plot_exercise_boundary(**TRAINING_CONTRACT, path=os.path.join(FIGURES_DIR, "exercise_boundary.png"))
    plot_nn_vs_binomial(df, path=os.path.join(FIGURES_DIR, "nn_vs_binomial.png"))
    print(f"figures saved to {FIGURES_DIR}")

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("Binomial vs. NN vs. RL -- summary\n" + "=" * 40 + "\n\n")
        f.write(f"NN vs binomial (n={len(df)}):\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"  intrinsic_violations: {intrinsic_violations} / {len(df)}\n")
        f.write(f"  spot_monotonicity_violations: {len(increases)}\n\n")
        f.write(f"RL, training contract {TRAINING_CONTRACT}:\n")
        f.write(rl_comparison.to_string(index=False) + "\n")
        f.write(f"  binomial price: {training_binomial_price:.4f}\n")
        f.write(f"  boundary agreement: {agreement:.4f}\n\n")
        f.write(f"RL, full grid (out-of-distribution check, n={len(contracts)}):\n")
        for k, v in rl_grid_summary.items():
            f.write(f"  {k}: {v:.4f}\n")

    print(f"\nall results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
