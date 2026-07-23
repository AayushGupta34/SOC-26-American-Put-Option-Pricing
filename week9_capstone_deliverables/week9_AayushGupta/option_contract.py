"""
A single consistent contract object used everywhere in the Week 9 comparison.

Every prior week (4, 6, 7, 8) passed S0, K, T, r, sigma, steps around as loose,
independently-ordered arguments. That's exactly the kind of mismatch risk the
Week 9 brief warns about: "a comparison is not fair if the NN uses one grid,
the RL policy uses another." OptionContract gives the binomial pricer, the NN
pricer, and the RL policy the same object to read from, so nobody can
accidentally pass e.g. (K, S0, ...) in the wrong order to just one of them.
"""

from dataclasses import dataclass


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
        assert self.S0 > 0, "S0 must be positive"
        assert self.K > 0, "K must be positive"
        assert self.T > 0, "T must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert self.steps >= 1, "steps must be >= 1"
        assert self.option_type in {"put", "call"}, "option_type must be 'put' or 'call'"
        return self

    def as_row(self):
        """[S0, K, T, r, sigma] in the exact order week6's PutPricerNet expects."""
        return [self.S0, self.K, self.T, self.r, self.sigma]
