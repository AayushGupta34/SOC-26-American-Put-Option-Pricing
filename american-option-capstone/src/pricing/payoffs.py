"""Payoff functions for European/American puts and calls.

Kept in their own module because three different things need the exact same
formula: the binomial tree's terminal/intrinsic values, the RL environment's
reward, and the sanity tests. Importing from one place means there's no
chance of the put payoff being written slightly differently in two places.
"""

import numpy as np


def put_payoff(S, K):
    """max(K - S, 0), works on scalars or numpy arrays."""
    return np.maximum(K - S, 0.0)


def call_payoff(S, K):
    """max(S - K, 0), works on scalars or numpy arrays."""
    return np.maximum(S - K, 0.0)


def payoff(S, K, option_type="put"):
    if option_type == "put":
        return put_payoff(S, K)
    if option_type == "call":
        return call_payoff(S, K)
    raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")
