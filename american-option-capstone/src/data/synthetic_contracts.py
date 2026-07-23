"""Everything about *what contracts we're pricing*, as opposed to how.

Two different things live here on purpose:

- `sample_contracts` draws random contracts for training the NN pricer --
  it needs broad, randomized coverage so the network doesn't just memorize
  a grid.
- `make_shared_grid` is the opposite: a small, fixed, deliberately-chosen set
  of contracts (spanning ITM/ATM/OTM, short/long maturity, low/high vol) that
  every comparison in evaluation/comparison.py runs against, so the binomial,
  NN, and RL numbers are always talking about the same contracts.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from pricing.binomial import crr_price


@dataclass(frozen=True)
class OptionContract:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    steps: int = 100
    option_type: str = "put"

    def validate(self):
        assert self.S0 > 0
        assert self.K > 0
        assert self.T > 0
        assert self.sigma > 0
        assert self.steps >= 1
        assert self.option_type in {"put", "call"}
        return self

    def as_row(self):
        """[S0, K, T, r, sigma], the order the NN pricer expects."""
        return [self.S0, self.K, self.T, self.r, self.sigma]


def sample_contracts(n: int, seed: int = 42) -> np.ndarray:
    """Random contracts for NN training -- returns an (n, 5) array of
    [S0, K, T, r, sigma], matching OptionContract.as_row()'s column order."""
    rng = np.random.default_rng(seed)
    S0 = rng.uniform(60.0, 140.0, size=n)
    K = rng.uniform(80.0, 120.0, size=n)
    T = rng.uniform(0.05, 2.0, size=n)
    r = rng.uniform(0.00, 0.10, size=n)
    sigma = rng.uniform(0.10, 0.50, size=n)
    return np.column_stack([S0, K, T, r, sigma])


def label_contracts(X: np.ndarray, steps: int = 500) -> np.ndarray:
    """Binomial American put price for each row of X -- these are the NN's
    training labels."""
    y = np.empty(len(X), dtype=np.float64)
    for i, (S0, K, T, r, sigma) in enumerate(X):
        y[i] = crr_price(S0, K, T, r, sigma, steps, option_type="put", american=True)
    return y


SHARED_GRID_SPOTS = [70, 80, 90, 100, 110, 120, 130]
SHARED_GRID_STRIKES = [100]
SHARED_GRID_MATURITIES = [0.25, 0.5, 1.0, 2.0]
SHARED_GRID_RATES = [0.02, 0.05]
SHARED_GRID_SIGMAS = [0.15, 0.25, 0.40]
SHARED_GRID_STEPS = 100


def make_shared_grid() -> list[OptionContract]:
    """The 168-contract grid every method-vs-method comparison runs
    against: deep ITM through deep OTM, short and long maturities, low
    and high vol. A single hand-picked contract can hide exactly the
    corners where a method breaks, so this deliberately spans them."""
    contracts = []
    for S0, K, T, r, sigma in itertools.product(
        SHARED_GRID_SPOTS, SHARED_GRID_STRIKES, SHARED_GRID_MATURITIES,
        SHARED_GRID_RATES, SHARED_GRID_SIGMAS,
    ):
        contracts.append(
            OptionContract(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=SHARED_GRID_STEPS).validate()
        )
    return contracts
