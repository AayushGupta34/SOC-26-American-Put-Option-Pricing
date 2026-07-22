import numpy as np
from part_b_environment import AmericanPutEnv

def discretize_state(state, n_time=20, n_money=30, money_min=0.5, money_max=1.5):
    time_fraction, moneyness = state
    t_bin = int(np.clip(time_fraction * n_time, 0, n_time - 1))
    m_scaled = (moneyness - money_min) / (money_max - money_min)
    m_bin = int(np.clip(m_scaled * n_money, 0, n_money - 1))
    return t_bin, m_bin


def train_q_learning(episodes=5000, alpha=0.05, epsilon=0.15,
                     n_time=20, n_money=30, seed=42):
    """Epsilon-greedy tabular Q-learning. Returns the learned Q-table."""
    n_actions = 2
    Q = np.zeros((n_time, n_money, n_actions), dtype=np.float64)
    env = AmericanPutEnv(seed=seed)

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            s_idx = discretize_state(state, n_time, n_money)
            if env.rng.random() < epsilon:
                action = int(env.rng.integers(0, n_actions))
            else:
                action = int(np.argmax(Q[s_idx]))
            next_state, reward, done, info = env.step_env(action)
            ns_idx = discretize_state(next_state, n_time, n_money)
            target = reward if done else reward + env.discount * np.max(Q[ns_idx])
            Q[s_idx + (action,)] += alpha * (target - Q[s_idx + (action,)])
            state = next_state
    return Q

def always_hold_policy(state):
    return AmericanPutEnv.HOLD


def immediate_exercise_policy(state):
    return AmericanPutEnv.EXERCISE


def make_random_policy(env):
    def policy(state):
        return int(env.rng.integers(0, 2))   # true 50/50
    return policy


def make_q_policy(Q, n_time=20, n_money=30):
    def policy(state):
        s_idx = discretize_state(state, n_time, n_money)
        return int(np.argmax(Q[s_idx]))
    return policy

def run_policy(env, policy_fn, episodes=1000):
    """Roll out policy_fn. Returns (rewards, exercise_steps)."""
    rewards = []
    exercise_steps = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy_fn(state)
            state, reward, done, info = env.step_env(action)
            total += reward
            if done and info["reason"] == "exercise":
                exercise_steps.append(env.step)
        rewards.append(total)
    return np.array(rewards), np.array(exercise_steps)


def tabulate_exercise_timing(name, exercise_steps, total_steps=50, n_bins=10):
    """Print a text histogram of WHEN early exercise happened."""
    print(f"\nExercise timing - {name} (n={len(exercise_steps)} early exercises):")
    if len(exercise_steps) == 0:
        print("  (no early exercises to tabulate)")
        return
    counts, edges = np.histogram(exercise_steps, bins=n_bins, range=(0, total_steps))
    max_count = counts.max()
    for i in range(n_bins):
        bar = "#" * int(40 * counts[i] / max_count) if max_count > 0 else ""
        print(f"  step {int(edges[i]):>2}-{int(edges[i + 1]):>2}: {counts[i]:>4}  {bar}")

if __name__ == "__main__":
    EPISODES = 1000

    print("=" * 54)
    print("PART C — policy comparison (%d episodes each)" % EPISODES)
    print("=" * 54)
    print(f"{'policy':<26}{'avg_reward':<14}{'exercise_rate':<14}")

    env = AmericanPutEnv(seed=100)
    r, ex = run_policy(env, always_hold_policy, EPISODES)
    print(f"{'always-hold-to-expiry':<26}{r.mean():<14.4f}{len(ex) / EPISODES:<14.3f}")

    env = AmericanPutEnv(seed=100)
    r, ex = run_policy(env, immediate_exercise_policy, EPISODES)
    print(f"{'immediate-exercise':<26}{r.mean():<14.4f}{len(ex) / EPISODES:<14.3f}")

    env = AmericanPutEnv(seed=100)
    r_rand, ex_rand = run_policy(env, make_random_policy(env), EPISODES)
    print(f"{'random':<26}{r_rand.mean():<14.4f}{len(ex_rand) / EPISODES:<14.3f}")
    Q = train_q_learning(episodes=5000, seed=42)
    env = AmericanPutEnv(seed=100)
    r_q, ex_q = run_policy(env, make_q_policy(Q), EPISODES)
    print(f"{'q-learning':<26}{r_q.mean():<14.4f}{len(ex_q) / EPISODES:<14.3f}")

    print("\n" + "=" * 54)
    print("PART C — exercise timing for nontrivial policies")
    print("=" * 54)
    tabulate_exercise_timing("random", ex_rand, total_steps=50)
    tabulate_exercise_timing("q-learning", ex_q, total_steps=50)
