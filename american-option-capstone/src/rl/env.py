"""The environment an RL agent interacts with to learn early exercise.

Deliberately built on the *same* risk-neutral u/d/p as the binomial tree
(pricing/binomial.py) for one fixed contract, so a policy that converges to
optimal here should match the binomial price for that contract. The state
never includes future prices -- only where we are in time and how far
in/out of the money we are right now -- so there's no way for the agent to
peek ahead.
"""

import math

import numpy as np


def make_state(step, steps, spot, strike):
    time_fraction = step / steps
    time_to_expiry = 1.0 - time_fraction
    moneyness = spot / strike
    return np.array([time_fraction, time_to_expiry, moneyness], dtype=np.float32)


class AmericanPutEnv:
    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50, seed=42):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.rng = np.random.default_rng(seed)

        self.dt = T / steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)
        self.reset()

    def _state(self):
        return make_state(self.step, self.steps, self.spot, self.K)

    def reset(self):
        self.step = 0
        self.spot = self.S0
        self.done = False
        return self._state()

    def step_env(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        payoff = max(self.K - self.spot, 0.0)

        if action == self.EXERCISE:
            self.done = True
            return self._state(), payoff, True, {"reason": "exercise"}

        if action != self.HOLD:
            raise ValueError("action must be 0=hold or 1=exercise")

        if self.rng.random() < self.p:
            self.spot *= self.u
        else:
            self.spot *= self.d

        self.step += 1
        if self.step >= self.steps:
            self.done = True
            terminal_payoff = max(self.K - self.spot, 0.0)
            return self._state(), terminal_payoff, True, {"reason": "expiry"}

        return self._state(), 0.0, False, {"reason": "hold"}


# The one contract the trained DQN in reports/results/dqn_online.pt actually
# learned. See rl/evaluate_policy.py and the report for why this matters.
TRAINING_CONTRACT = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)


def env_factory(seed=42, **overrides):
    params = {**TRAINING_CONTRACT, **overrides}
    return AmericanPutEnv(seed=seed, **params)
