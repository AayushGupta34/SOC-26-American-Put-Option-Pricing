"""
Ties the CRR binomial pricer (Week 4), the NN pricer (Week 6), and the RL
stopping policy (Weeks 7-8) into one set of comparison tables.

Run from the week9_AayushGupta/ folder:

    python compare_all.py

Outputs (all written to week9_AayushGupta/output/):
    pricing_comparison_grid.csv   - one row per contract: binomial vs NN price
    bucket_summary_moneyness.csv
    bucket_summary_maturity.csv
    bucket_summary_volatility.csv
    rl_policy_comparison.csv
    summary.txt                  - headline numbers, human-readable
"""

import os

import numpy as np
import pandas as pd

from contract_grid import make_contract_objects
from option_contract import OptionContract
from pricers import BinomialAmericanPutPricer
from nn_pricer import NNPricer
from pricing_metrics import pricing_metrics, count_intrinsic_violations, put_spot_monotonicity_check
from rl_policy import build_policy_comparison, boundary_agreement, WEEK8_CONTRACT, load_trained_dqn
from boundary_utils import compute_boundary_by_step
import week8  # for make_dqn_policy / AmericanPutEnv, via rl_policy's sys.path setup

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def compare_one_contract(contract, binomial_pricer, nn_pricer):
    binomial_result = binomial_pricer.price(contract)
    nn_price = nn_pricer.predict(contract)
    return {
        "S0": contract.S0, "K": contract.K, "T": contract.T,
        "r": contract.r, "sigma": contract.sigma, "steps": contract.steps,
        "binomial_price": binomial_result.price,
        "nn_price": nn_price,
        "nn_error": nn_price - binomial_result.price,
    }


def add_slicing_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ratio = df["S0"] / df["K"]
    df["moneyness_bucket"] = pd.cut(
        ratio, bins=[0.0, 0.9, 1.1, 10.0], labels=["ITM put", "ATM", "OTM put"]
    )
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building shared contract grid...")
    contracts = make_contract_objects()
    print(f"  {len(contracts)} contracts")

    print("Loading binomial pricer (Week 4) and NN pricer (Week 6)...")
    binomial_pricer = BinomialAmericanPutPricer()
    nn_pricer = NNPricer()

    print("Pricing every contract with both methods...")
    rows = [compare_one_contract(c, binomial_pricer, nn_pricer) for c in contracts]
    df = pd.DataFrame(rows)
    df = add_slicing_columns(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "pricing_comparison_grid.csv"), index=False)

    overall = pricing_metrics(df["binomial_price"], df["nn_price"])
    intrinsic_violations = count_intrinsic_violations(df)

    print("\n=== Overall NN vs Binomial accuracy (full grid, n={}) ===".format(len(df)))
    for k, v in overall.items():
        print(f"  {k:22s}: {v:.6f}")
    print(f"  {'intrinsic_violations':22s}: {intrinsic_violations} / {len(df)}")

    print("\n=== Sliced summaries ===")
    bucket_tables = {}
    for by_col, out_name in [
        ("moneyness_bucket", "bucket_summary_moneyness.csv"),
        ("maturity_bucket", "bucket_summary_maturity.csv"),
        ("vol_bucket", "bucket_summary_volatility.csv"),
    ]:
        summary = summarize_by(df, by_col)
        bucket_tables[by_col] = summary
        summary.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False)
        print(f"\n-- by {by_col} --")
        print(summary.to_string(index=False))

    print("\n=== NN sanity check: put price vs spot monotonicity (K=100,T=1,r=5%,sigma=25%) ===")
    increases = put_spot_monotonicity_check(
        lambda S0, K, T, r, sigma: nn_pricer.predict(
            OptionContract(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=100)
        )
    )
    print(f"  monotonicity violations: {len(increases)} (put price should be non-increasing in spot)")

    print("\n=== RL policy comparison (single contract week8 was trained on: "
          f"S0={WEEK8_CONTRACT.S0}, K={WEEK8_CONTRACT.K}, T={WEEK8_CONTRACT.T}, "
          f"r={WEEK8_CONTRACT.r}, sigma={WEEK8_CONTRACT.sigma}, steps={WEEK8_CONTRACT.steps}) ===")
    rl_comparison = build_policy_comparison(episodes=10_000)
    rl_comparison.to_csv(os.path.join(OUTPUT_DIR, "rl_policy_comparison.csv"), index=False)
    print(rl_comparison.to_string(index=False))

    week8_binomial_price = binomial_pricer.price(WEEK8_CONTRACT).price
    print(f"\n  binomial (American) benchmark for this contract: {week8_binomial_price:.4f}")

    print("\n=== RL boundary agreement (DQN decisions vs binomial-optimal exercise) ===")
    online = load_trained_dqn()
    boundary_by_step = compute_boundary_by_step(WEEK8_CONTRACT)
    agreement = boundary_agreement(
        week8.make_dqn_policy(online), boundary_by_step,
        steps=WEEK8_CONTRACT.steps, K=WEEK8_CONTRACT.K,
    )
    print(f"  agreement fraction: {agreement:.4f}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("Week 9 Capstone -- Unified Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Contracts evaluated (binomial vs NN): {len(df)}\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"  intrinsic_violations: {intrinsic_violations} / {len(df)}\n")
        f.write(f"  spot-monotonicity violations: {len(increases)}\n\n")
        f.write(f"RL contract (week8): S0={WEEK8_CONTRACT.S0} K={WEEK8_CONTRACT.K} "
                f"T={WEEK8_CONTRACT.T} r={WEEK8_CONTRACT.r} sigma={WEEK8_CONTRACT.sigma} "
                f"steps={WEEK8_CONTRACT.steps}\n")
        f.write(f"  binomial (American) price: {week8_binomial_price:.4f}\n")
        f.write(rl_comparison.to_string(index=False) + "\n")
        f.write(f"  boundary agreement (DQN vs binomial-optimal): {agreement:.4f}\n")
        f.write("\nNOTE: the RL comparison above is single-contract -- dqn_online.pt was "
                "trained only against this one contract's risk-neutral dynamics, not the "
                "full grid. See reports/ for the full limitations discussion.\n")

    print(f"\nAll tables written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
