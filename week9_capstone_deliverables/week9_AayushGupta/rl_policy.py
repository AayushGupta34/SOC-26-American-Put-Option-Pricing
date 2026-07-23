"""
RL policy validation, reusing Week 8's already-trained DQN
(week8_AayushGupta/dqn_online.pt) rather than retraining.

IMPORTANT SCOPE NOTE: week8's AmericanPutEnv is hard-coded to one specific
contract (S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=50 -- see the
module-level `S0, K, T, r, sigma, steps = ...` line in week8_AayushGupta/
week8.py) and dqn_online.pt was trained *only* against that contract's
risk-neutral dynamics. It was never trained to generalize across the full
Week 9 contract grid. So unlike the NN pricer (which can be evaluated across
every contract in contract_grid.py), the RL comparison in this file is
necessarily a single-contract validation: "does the trained policy's value
and exercise behavior match the binomial optimum for the one contract it was
trained on?" This should be called out explicitly in the final report's
limitations section.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

from option_contract import OptionContract

_WEEK8_DIR = os.path.join(os.path.dirname(__file__), "..", "week8_AayushGupta")
if _WEEK8_DIR not in sys.path:
    sys.path.insert(0, _WEEK8_DIR)

import week8  # noqa: E402  (must come after sys.path insert)

# The exact contract week8's environment and dqn_online.pt were built for.
WEEK8_CONTRACT = OptionContract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)


def load_trained_dqn(checkpoint_path: str = None):
    checkpoint_path = checkpoint_path or os.path.join(_WEEK8_DIR, "dqn_online.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. Run week8_AayushGupta/week8.py "
            f"first (it now requires `python week8.py` explicitly, since it no "
            f"longer trains automatically on import) to generate dqn_online.pt."
        )
    model = week8.QNetwork(state_dim=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def build_policy_comparison(episodes: int = 10_000) -> pd.DataFrame:
    """Compare always-hold, immediate-exercise, random, and the trained DQN,
    all against week8's own evaluate_policy, all on WEEK8_CONTRACT."""
    online = load_trained_dqn()

    policies = {
        "always-hold-to-expiry": week8.always_hold_policy,
        "immediate-exercise": week8.immediate_exercise_policy,
        "random": week8.make_random_policy(np.random.default_rng(42)),
        "dqn": week8.make_dqn_policy(online),
    }

    rows = []
    for name, policy_fn in policies.items():
        result = week8.evaluate_policy(week8.env_factory, policy_fn, episodes=episodes)
        rows.append({
            "policy": name,
            "value": result["value"],
            "std_error": result["std_error"],
            "exercise_rate": result["exercise_rate"],
            "avg_exercise_step": result["avg_exercise_step"],
        })
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def boundary_agreement(policy_fn, boundary_by_step: dict, steps: int = 50, K: float = 100.0) -> float:
    """Fraction of (step, moneyness) grid points where the RL policy's
    hold/exercise decision agrees with the binomial-optimal decision implied
    by the exercise boundary."""
    checks = []
    money_grid = np.linspace(0.6, 1.4, 81)
    for step in range(steps):
        boundary_spot = boundary_by_step.get(step)
        if boundary_spot is None:
            continue
        for m in money_grid:
            S = m * K
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            policy_exercise = policy_fn(state) == week8.AmericanPutEnv.EXERCISE
            binomial_exercise = S <= boundary_spot
            checks.append(policy_exercise == binomial_exercise)
    return float(np.mean(checks)) if checks else None
