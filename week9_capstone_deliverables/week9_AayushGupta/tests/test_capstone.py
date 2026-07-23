"""
Week 9 capstone tests.

week4_AayushGupta/test_american_put.py already covers American >= European
value and monotonicity in spot/volatility for the raw `crr_put_price`
function directly -- that file is left untouched and still passes on its own
(`python week4_AayushGupta/test_american_put.py`).

This file adds two things Week 9 specifically calls for:
  1. A payoff-function test: put payoff is max(K-S, 0), never negative.
  2. An RL environment terminal-behavior test: episodes actually end (no
     infinite loops) and reward is exactly zero everywhere except the
     terminal step -- i.e. the agent can't get paid for merely holding.

It also re-runs the American >= European and monotonicity checks, but through
the Week 9 `BinomialAmericanPutPricer` wrapper (pricers.py) + `OptionContract`
instead of calling `crr_put_price` directly, so the test suite validates the
actual Week 9 integration layer, not just the underlying week4 function.

Run with:
    cd week9_AayushGupta && python -m pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from option_contract import OptionContract
from pricers import BinomialAmericanPutPricer
from rl_policy import load_trained_dqn
import week8


def test_american_not_less_than_european_via_wrapper():
    # OptionContract/BinomialAmericanPutPricer only ever price American-style
    # (option_type toggles put/call, not American/European), so the European
    # comparison price is computed directly via crr_put_price(american=False).
    pricer = BinomialAmericanPutPricer()
    contract = OptionContract(S0=100, K=105, T=1.0, r=0.05, sigma=0.25, steps=500).validate()
    from american_put import crr_put_price
    euro = crr_put_price(contract.S0, contract.K, contract.T, contract.r, contract.sigma,
                          contract.steps, american=False)
    amer = pricer.price(contract).price
    assert amer >= euro


def test_put_value_falls_as_spot_rises_via_wrapper():
    pricer = BinomialAmericanPutPricer()
    low_spot = pricer.price(OptionContract(80, 100, 1.0, 0.05, 0.25, 500).validate()).price
    high_spot = pricer.price(OptionContract(120, 100, 1.0, 0.05, 0.25, 500).validate()).price
    assert low_spot > high_spot


def test_put_payoff_never_negative():
    """Payoff is max(K - S, 0) -- must never go negative, for any S including
    S far above K."""
    K = 100.0
    spots = np.linspace(0.0, 300.0, 50)
    payoffs = np.maximum(K - spots, 0.0)
    assert np.all(payoffs >= 0.0)

    # Also check it through the actual RL environment's own payoff logic,
    # not just a reimplementation, by exercising immediately from a range
    # of starting spots.
    for S0 in [50.0, 100.0, 150.0, 250.0]:
        env = week8.AmericanPutEnv(S0=S0, K=K, T=1.0, r=0.05, sigma=0.25, steps=10, seed=1)
        env.reset()
        _, reward, done, info = env.step_env(env.EXERCISE)
        assert reward >= 0.0
        assert done
        assert info["reason"] == "exercise"


def test_rl_env_terminates_and_reward_is_zero_except_terminal_step():
    """Random-policy episodes must terminate within `steps` HOLD actions
    (plus the terminating action), and reward must be 0.0 on every
    non-terminal step -- confirms the agent can't collect reward just by
    holding, which would let it exploit a training bug."""
    rng = np.random.default_rng(123)
    steps = 25

    for seed in range(20):
        env = week8.AmericanPutEnv(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25,
                                    steps=steps, seed=seed)
        env.reset()
        done = False
        n_actions = 0
        max_actions = steps + 1  # at most `steps` holds, plus the terminating action

        while not done:
            action = int(rng.integers(0, 2))
            _, reward, done, info = env.step_env(action)
            n_actions += 1

            if not done:
                assert reward == 0.0, "non-terminal steps must pay zero reward"
            else:
                assert reward >= 0.0
                assert info["reason"] in {"exercise", "expiry"}

            assert n_actions <= max_actions, "episode did not terminate in time -- possible infinite loop"


def test_dqn_checkpoint_loads_and_produces_valid_actions():
    """The trained DQN checkpoint should load without shape errors and always
    return a valid action (0 or 1) for a range of states."""
    model = load_trained_dqn()
    for step in [0, 10, 25, 49]:
        for moneyness in [0.7, 1.0, 1.3]:
            state = np.array([step / 50, 1.0 - step / 50, moneyness], dtype=np.float32)
            action = week8.greedy_action(model, state)
            assert action in (0, 1)
