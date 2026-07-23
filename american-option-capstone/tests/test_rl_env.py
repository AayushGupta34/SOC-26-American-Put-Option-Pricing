import numpy as np

from rl.env import AmericanPutEnv


def test_exercise_pays_nonnegative_and_ends_episode():
    for S0 in [50.0, 100.0, 150.0]:
        env = AmericanPutEnv(S0=S0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=10, seed=1)
        env.reset()
        _, reward, done, info = env.step_env(env.EXERCISE)
        assert reward >= 0.0
        assert done
        assert info["reason"] == "exercise"


def test_cannot_step_after_done():
    env = AmericanPutEnv(seed=1)
    env.reset()
    env.step_env(env.EXERCISE)
    try:
        env.step_env(env.HOLD)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_episode_terminates_and_reward_is_zero_except_terminal_step():
    rng = np.random.default_rng(123)
    steps = 25

    for seed in range(20):
        env = AmericanPutEnv(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=steps, seed=seed)
        env.reset()
        done = False
        n_actions = 0
        max_actions = steps + 1

        while not done:
            action = int(rng.integers(0, 2))
            _, reward, done, info = env.step_env(action)
            n_actions += 1

            if not done:
                assert reward == 0.0
            else:
                assert reward >= 0.0
                assert info["reason"] in {"exercise", "expiry"}

            assert n_actions <= max_actions, "episode did not terminate -- possible infinite loop"


def test_expiry_pays_intrinsic_value():
    env = AmericanPutEnv(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=5, seed=1)
    env.reset()
    done = False
    reward = None
    info = None
    while not done:
        _, reward, done, info = env.step_env(env.HOLD)
    assert info["reason"] == "expiry"
    assert reward == max(env.K - env.spot, 0.0)
