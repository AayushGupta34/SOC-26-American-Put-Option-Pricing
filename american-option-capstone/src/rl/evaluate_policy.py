"""Evaluate the trained DQN against the fixed baselines and the binomial
exercise boundary.

    python -m rl.evaluate_policy

Everything here is read-only with respect to the checkpoint -- it loads
reports/results/dqn_online.pt and never retrains.
"""

import os

import numpy as np
import pandas as pd
import torch

from pricing.binomial import crr_price, exercise_boundary_by_step
from rl.env import AmericanPutEnv, env_factory, TRAINING_CONTRACT
from rl.train_dqn import QNetwork, CHECKPOINT_PATH


def always_hold_policy(state):
    return AmericanPutEnv.HOLD


def immediate_exercise_policy(state):
    return AmericanPutEnv.EXERCISE


def make_random_policy(rng):
    def policy(state):
        return int(rng.integers(0, 2))
    return policy


def make_dqn_policy(model):
    def policy(state):
        with torch.no_grad():
            q = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())
    return policy


def load_trained_dqn(checkpoint_path: str = CHECKPOINT_PATH) -> QNetwork:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `python -m rl.train_dqn` first."
        )
    model = QNetwork(state_dim=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_policy(factory, policy_fn, episodes=10_000):
    discounted_rewards = []
    exercise_steps = []
    for seed in range(episodes):
        env = factory(seed=seed)
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = policy_fn(state)
            state, reward, done, info = env.step_env(action)
            if done:
                discounted_rewards.append((env.discount ** step) * reward)
                if info["reason"] == "exercise":
                    exercise_steps.append(step)
            step += 1
    return {
        "value": float(np.mean(discounted_rewards)),
        "std_error": float(np.std(discounted_rewards) / np.sqrt(episodes)),
        "exercise_rate": len(exercise_steps) / episodes,
        "avg_exercise_step": float(np.mean(exercise_steps)) if exercise_steps else None,
    }


def build_policy_comparison(episodes: int = 10_000) -> pd.DataFrame:
    """Value/exercise-rate comparison on the DQN's own training contract."""
    online = load_trained_dqn()
    policies = {
        "always-hold-to-expiry": always_hold_policy,
        "immediate-exercise": immediate_exercise_policy,
        "random": make_random_policy(np.random.default_rng(42)),
        "dqn": make_dqn_policy(online),
    }
    rows = []
    for name, policy_fn in policies.items():
        result = evaluate_policy(env_factory, policy_fn, episodes=episodes)
        rows.append({"policy": name, **result})
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def boundary_agreement(policy_fn, steps: int = None, K: float = None) -> float:
    """Fraction of (step, moneyness) states where the policy's decision
    matches the binomial-optimal exercise/hold decision."""
    steps = steps or TRAINING_CONTRACT["steps"]
    K = K or TRAINING_CONTRACT["K"]
    boundary = exercise_boundary_by_step(**TRAINING_CONTRACT)

    checks = []
    for step in range(steps):
        boundary_spot = boundary.get(step)
        if boundary_spot is None:
            continue
        for m in np.linspace(0.6, 1.4, 81):
            S = m * K
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            policy_exercise = policy_fn(state) == AmericanPutEnv.EXERCISE
            binomial_exercise = S <= boundary_spot
            checks.append(policy_exercise == binomial_exercise)
    return float(np.mean(checks)) if checks else None


def evaluate_dqn_across_grid(contracts, episodes: int = 300) -> pd.DataFrame:
    """Out-of-distribution check: runs the trained DQN (no retraining)
    against every contract in the shared grid, not just TRAINING_CONTRACT.
    The DQN has only ever seen TRAINING_CONTRACT's dynamics, so don't expect
    this to look as good as build_policy_comparison() above -- that's the
    point of running it."""
    online = load_trained_dqn()
    dqn_policy = make_dqn_policy(online)

    rows = []
    for c in contracts:
        def factory(seed, c=c):
            return AmericanPutEnv(S0=c.S0, K=c.K, T=c.T, r=c.r, sigma=c.sigma, steps=c.steps, seed=seed)

        dqn_result = evaluate_policy(factory, dqn_policy, episodes=episodes)
        hold_result = evaluate_policy(factory, always_hold_policy, episodes=episodes)
        binomial_price = crr_price(c.S0, c.K, c.T, c.r, c.sigma, c.steps, option_type="put", american=True)

        rows.append({
            "S0": c.S0, "K": c.K, "T": c.T, "r": c.r, "sigma": c.sigma, "steps": c.steps,
            "dqn_value": dqn_result["value"],
            "dqn_exercise_rate": dqn_result["exercise_rate"],
            "hold_value": hold_result["value"],
            "binomial_price": binomial_price,
            "dqn_gap_vs_binomial": dqn_result["value"] - binomial_price,
        })
    return pd.DataFrame(rows)


def main():
    print("=== DQN vs. baselines, on its own training contract ===")
    print(build_policy_comparison().to_string(index=False))

    online = load_trained_dqn()
    agreement = boundary_agreement(make_dqn_policy(online))
    print(f"\nboundary agreement: {agreement:.4f}")


if __name__ == "__main__":
    main()
