"""Black-Scholes closed-form price for a European put or call.

There's no American closed-form solution (that's the whole reason this
project exists), but Black-Scholes gives a fast, exact check on the
*European* side: as steps -> large in the binomial tree, its European price
should converge to this. `tests/test_binomial.py` uses exactly that as a
consistency check, independent of relying on the binomial tree to grade its
own convergence.
"""

import math

from pricing.payoffs import payoff


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S0, K, T, r, sigma, option_type="put"):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return float(payoff(S0, K, option_type))
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
    raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")
