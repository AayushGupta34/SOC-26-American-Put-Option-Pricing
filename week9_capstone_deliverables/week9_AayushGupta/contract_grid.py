"""
The single shared evaluation grid. Every comparison in Week 9 (binomial vs NN,
DQN vs binomial exercise boundary) is run against this same grid, so results
are comparable across methods.

The grid deliberately spans deep ITM, ATM, and deep OTM puts, short and long
maturities, and low/high volatility -- a single hand-picked contract can't
reveal where a method breaks, which is exactly what the Week 9 brief calls out.
"""

import itertools

import pandas as pd

from option_contract import OptionContract

SPOTS = [70, 80, 90, 100, 110, 120, 130]
STRIKES = [100]
MATURITIES = [0.25, 0.5, 1.0, 2.0]
RATES = [0.02, 0.05]
SIGMAS = [0.15, 0.25, 0.40]
STEPS = 100


def make_contract_grid() -> pd.DataFrame:
    rows = []
    for S0, K, T, r, sigma in itertools.product(SPOTS, STRIKES, MATURITIES, RATES, SIGMAS):
        rows.append({"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "steps": STEPS})
    return pd.DataFrame(rows)


def make_contract_objects() -> list[OptionContract]:
    df = make_contract_grid()
    return [
        OptionContract(S0=row.S0, K=row.K, T=row.T, r=row.r, sigma=row.sigma, steps=int(row.steps)).validate()
        for row in df.itertuples(index=False)
    ]


if __name__ == "__main__":
    grid = make_contract_grid()
    print(f"Grid size: {len(grid)} contracts")
    print(grid.head())
