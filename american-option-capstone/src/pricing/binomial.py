"""Cox-Ross-Rubinstein binomial tree, American or European, put or call.

This is the same tree from the earlier weekly submission (american_put.py),
generalized slightly: it now takes an `option_type` instead of being
put-only, and shares its payoff formula with pricing/payoffs.py instead of
recomputing max(K-S,0) inline. The actual backward-induction logic hasn't
changed.
"""

import math

import numpy as np

from pricing.payoffs import payoff


def crr_price(S0, K, T, r, sigma, steps, option_type="put", american=True):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return float(payoff(S0, K, option_type))
    if sigma <= 0:
        raise ValueError("sigma must be positive for the CRR model")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")

    steps = int(steps)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)

    if not (0.0 < p < 1.0):
        raise ValueError("Invalid risk-neutral probability; increase steps or check inputs")

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = payoff(stock, K, option_type)

    for i in range(steps - 1, -1, -1):
        value = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])

        if american:
            j = np.arange(i + 1)
            stock = S0 * (u ** j) * (d ** (i - j))
            exercise = payoff(stock, K, option_type)
            value = np.maximum(value, exercise)

    return float(value[0])


def exercise_boundary_by_step(S0, K, T, r, sigma, steps, option_type="put"):
    """The highest (put) / lowest (call) stock price at each step where
    immediate exercise beats holding, as a {step: boundary_price} dict.

    Only meaningful for American options -- there's no early-exercise
    boundary for a European contract. Used by evaluation/plots.py to draw
    the exercise-boundary figure and by rl/evaluate_policy.py to check the
    DQN's decisions against the binomial-optimal ones.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = payoff(stock, K, option_type)

    boundary = {}

    for i in range(steps - 1, -1, -1):
        continuation = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])
        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = payoff(stock, K, option_type)
        exercise_now = exercise > continuation + 1e-10

        if np.any(exercise_now):
            if option_type == "put":
                boundary[i] = float(np.max(stock[exercise_now]))
            else:
                boundary[i] = float(np.min(stock[exercise_now]))

        value = np.maximum(continuation, exercise)

    return boundary
