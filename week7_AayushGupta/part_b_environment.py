import math
import numpy as np


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
        return np.array([self.step / self.steps, self.spot / self.K], dtype=np.float32)

    def reset(self):
        self.step = 0
        self.spot = self.S0
        self.done = False
        return self._state()

    def step_env(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")

        payoff = max(self.K - self.spot, 0.0)

        # EXERCISE terminates the episode immediately.
        if action == self.EXERCISE:
            self.done = True
            return self._state(), payoff, True, {"reason": "exercise"}

        if action != self.HOLD:
            raise ValueError("action must be 0=hold or 1=exercise")

        # HOLD: advance one risk-neutral binomial step.
        if self.rng.random() < self.p:
            self.spot *= self.u
        else:
            self.spot *= self.d

        self.step += 1

        # EXPIRY also terminates the episode.
        if self.step >= self.steps:
            self.done = True
            terminal_payoff = max(self.K - self.spot, 0.0)
            return self._state(), terminal_payoff, True, {"reason": "expiry"}

        return self._state(), 0.0, False, {"reason": "hold"}


# ----------------------------------------------------------------------
# Required invariant tests (Part B, item 4)
# ----------------------------------------------------------------------

def test_payoff_nonnegative():
    """Exercise reward is never negative (payoff is max(K - S, 0))."""
    env = AmericanPutEnv(seed=1)
    env.reset()
    _, reward, done, _ = env.step_env(env.EXERCISE)
    assert reward >= 0
    assert done


def test_cannot_step_after_done():
    """Stepping a finished episode must raise (prevents double exercise)."""
    env = AmericanPutEnv(seed=1)
    env.reset()
    env.step_env(env.EXERCISE)
    try:
        env.step_env(env.HOLD)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


# ----------------------------------------------------------------------
# Five sample episodes (Part B, item 5)
# ----------------------------------------------------------------------

def run_sample_episodes(n_episodes=5, seed=7, hold_bias=0.85):
    """
    Run n sample episodes and show the exercise/expiry reason for each.

    NOTE: a pure 50/50 random action on an at-the-money contract almost always
    draws EXERCISE on the first step and ends instantly at zero payoff, which
    demonstrates nothing. These demo episodes therefore bias the draw toward
    HOLD (hold_bias) so they actually visit the hold / expiry / in-the-money
    exercise branches. Part C's random-policy comparison still uses true 50/50.
    """
    env = AmericanPutEnv(seed=seed)
    print(f"\n{'ep':<4}{'reason':<12}{'stop_step':<12}{'reward':<10}{'S/K':<8}")
    print("-" * 48)
    for ep in range(n_episodes):
        env.reset()
        while True:
            action = env.HOLD if env.rng.random() < hold_bias else env.EXERCISE
            _, reward, done, info = env.step_env(action)
            if done:
                moneyness = env.spot / env.K
                print(f"{ep:<4}{info['reason']:<12}{env.step:<12}"
                      f"{round(reward, 4):<10}{round(moneyness, 4):<8}")
                break


if __name__ == "__main__":
    print("=" * 48)
    print("PART B — environment invariant tests")
    print("=" * 48)
    for name, fn in [("test_payoff_nonnegative", test_payoff_nonnegative),
                     ("test_cannot_step_after_done", test_cannot_step_after_done)]:
        try:
            fn()
            print(f"{name:<30} PASS")
        except AssertionError as e:
            print(f"{name:<30} FAIL: {e}")

    print("\n" + "=" * 48)
    print("PART B — 5 sample episodes (hold-biased demo, seed=7)")
    print("=" * 48)
    run_sample_episodes(5, seed=7)
