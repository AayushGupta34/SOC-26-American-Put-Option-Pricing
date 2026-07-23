import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from american_put import crr_put_price

S0, K, T, r, sigma, steps = 100.0, 100.0, 1.0, 0.05, 0.25, 50


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_state(step, steps, spot, strike):
    time_fraction = step / steps
    time_to_expiry = 1.0 - time_fraction
    moneyness = spot / strike
    return np.array([time_fraction, time_to_expiry, moneyness], dtype=np.float32)


class AmericanPutEnv:
    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0=S0, K=K, T=T, r=r, sigma=sigma, steps=steps, seed=42):
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


def env_factory(seed=42):
    return AmericanPutEnv(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=steps, seed=seed)


class QNetwork(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def greedy_action(model, state):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_t)
        return int(torch.argmax(q_values, dim=1).item())


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def compute_dqn_loss(online, target, batch, discount):
    states, actions, rewards, next_states, dones = batch

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_selected = online(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        q_target = rewards + (1.0 - dones) * discount * next_q

    return F.smooth_l1_loss(q_selected, q_target)


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
        return greedy_action(model, state)
    return policy


set_seeds(42)

env = env_factory(seed=42)

online = QNetwork(state_dim=3)
target = QNetwork(state_dim=3)
target.load_state_dict(online.state_dict())

optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
buffer = ReplayBuffer(capacity=50_000)

batch_size = 128
episodes = 10_000
target_update_every = 250
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999
updates = 0

log_every = 500
loss_ma = None
n_exercise = 0
n_expiry = 0
win_payoff, win_ex, win_step = [], 0, []

print(f"S0={S0} K={K} T={T} r={r} sigma={sigma} steps={steps} seed=42")
print(f"episodes={episodes} batch={batch_size} lr=1e-3 target_update={target_update_every}")

for episode in range(episodes):
    state = env.reset()
    done = False
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))

    while not done:
        if env.rng.random() < epsilon:
            action = int(env.rng.integers(0, 2))
        else:
            action = greedy_action(online, state)

        next_state, reward, done, info = env.step_env(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            loss = compute_dqn_loss(online, target, batch, env.discount)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
            optimizer.step()
            updates += 1

            l = float(loss.item())
            loss_ma = l if loss_ma is None else 0.99 * loss_ma + 0.01 * l

            if updates % target_update_every == 0:
                target.load_state_dict(online.state_dict())

    win_payoff.append((env.discount ** env.step) * reward)
    if info["reason"] == "exercise":
        n_exercise += 1
        win_ex += 1
        win_step.append(env.step)
    else:
        n_expiry += 1

    if (episode + 1) % log_every == 0:
        print(f"ep {episode+1:>6} payoff {np.mean(win_payoff):>7.4f} "
              f"eps {epsilon:>5.3f} loss_ma {loss_ma:>8.5f} "
              f"ex_rate {win_ex/len(win_payoff):>5.3f} "
              f"avg_ex_step {np.mean(win_step) if win_step else float('nan'):>5.1f}")
        win_payoff, win_ex, win_step = [], 0, []

torch.save(online.state_dict(), "dqn_online.pt")
print(f"terminations: exercise={n_exercise} expiry={n_expiry}")


def evaluate_policy(env_factory, policy_fn, episodes=10_000):
    discounted_rewards = []
    exercise_steps = []

    for seed in range(episodes):
        env = env_factory(seed=seed)
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


def policy_grid(policy_fn, steps=50, money_min=0.5, money_max=1.5, n_money=101):
    grid = []
    for step in range(steps + 1):
        row = []
        for m in np.linspace(money_min, money_max, n_money):
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            row.append(policy_fn(state))
        grid.append(row)
    return np.array(grid)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


eval_episodes = 5_000
benchmark = crr_put_price(S0, K, T, r, sigma, steps, american=True)
european = crr_put_price(S0, K, T, r, sigma, steps, american=False)

policies = [
    ("always-hold-to-expiry", always_hold_policy),
    ("immediate-exercise", immediate_exercise_policy),
    ("random", make_random_policy(np.random.default_rng(42))),
    ("dqn", make_dqn_policy(online)),
]

print(f"\n{'policy':<24}{'value':>10}{'std_err':>10}{'ex_rate':>10}{'avg_ex_step':>13}")
for name, fn in policies:
    m = evaluate_policy(env_factory, fn, eval_episodes)
    a = "n/a" if m["avg_exercise_step"] is None else f"{m['avg_exercise_step']:.2f}"
    print(f"{name:<24}{m['value']:>10.4f}{m['std_error']:>10.4f}"
          f"{m['exercise_rate']:>10.3f}{a:>13}")
print(f"{'binomial (American)':<24}{benchmark:>10.4f}")
print(f"{'binomial (European)':<24}{european:>10.4f}")

grid = policy_grid(make_dqn_policy(online), steps=steps)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="Greens",
               extent=[0.0, 1.0, 0.5, 1.5], vmin=0, vmax=1)
ax.axhline(1.0, color="0.25", ls="--", lw=1, label="at-the-money")
ax.set_xlabel("time fraction t/T")
ax.set_ylabel("moneyness S/K")
ax.set_title("Learned exercise region (green = EXERCISE)")
ax.legend(loc="upper right", fontsize=9)
fig.colorbar(im, ax=ax, label="exercise (1) / hold (0)")
fig.tight_layout()
fig.savefig("exercise_region.png", dpi=150)
plt.close(fig)
print("saved exercise_region.png")
