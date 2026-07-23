"""
Exercise-boundary extraction for the CRR binomial tree.

week4_AayushGupta/week4_report.py already computes this via
`crr_put_with_boundary`, but that function lives inline in a plotting script
(which also runs a full price-surface sweep and pops a matplotlib window at
import time) and returns a `[(time, boundary_stock), ...]` list meant only for
plotting. Week 9's `boundary_agreement` check (see rl_policy.py) needs the
boundary indexed *by step number* so it can be looked up against a state's
`time_fraction`. `compute_boundary_by_step` below is the same backward-
induction logic, refactored into a standalone, side-effect-free function that
returns a `{step: boundary_stock}` dict.
"""

import math

import numpy as np

from option_contract import OptionContract


def compute_boundary_by_step(contract: OptionContract) -> dict:
    S0, K, T, r, sigma, steps = contract.S0, contract.K, contract.T, contract.r, contract.sigma, contract.steps

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)

    boundary_by_step = {}

    for i in range(steps - 1, -1, -1):
        continuation = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])
        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = np.maximum(K - stock, 0.0)
        exercise_now = exercise > continuation + 1e-10

        if np.any(exercise_now):
            boundary_by_step[i] = float(np.max(stock[exercise_now]))

        value = np.maximum(continuation, exercise)

    return boundary_by_step
