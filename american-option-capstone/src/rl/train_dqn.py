"""Train the DQN early-exercise policy for the one contract in rl/env.py's
TRAINING_CONTRACT.

    python -m rl.train_dqn

Saves the trained weights to reports/results/dqn_online.pt (already
committed in this repo, so you only need to rerun this if you want to
retrain from scratch -- takes a couple of minutes for 10,000 episodes).
"""

import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.env import AmericanPutEnv, env_factory

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "reports", "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "reports", "figures")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "dqn_online.pt")


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

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


def main():
    set_seeds(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    env = env_factory(seed=42)
    online = QNetwork(state_dim=3)
    target = QNetwork(state_dim=3)
    target.load_state_dict(online.state_dict())

    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50_000)

    batch_size = 128
    episodes = 10_000
    target_update_every = 250
    epsilon_min = 0.05
    epsilon_decay = 0.999
    updates = 0
    log_every = 500
    loss_ma = None
    win_payoff, win_ex, win_step = [], 0, []

    print(f"episodes={episodes} batch={batch_size} lr=1e-3 target_update={target_update_every}")

    for episode in range(episodes):
        state = env.reset()
        done = False
        epsilon = max(epsilon_min, 1.0 * (epsilon_decay ** episode))

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
            win_ex += 1
            win_step.append(env.step)

        if (episode + 1) % log_every == 0:
            print(f"ep {episode + 1:>6} payoff {np.mean(win_payoff):>7.4f} "
                  f"eps {epsilon:>5.3f} loss_ma {loss_ma:>8.5f} "
                  f"ex_rate {win_ex / len(win_payoff):>5.3f}")
            win_payoff, win_ex, win_step = [], 0, []

    torch.save(online.state_dict(), CHECKPOINT_PATH)
    print(f"saved {CHECKPOINT_PATH}")

    grid = []
    for step in range(env.steps + 1):
        row = []
        for m in np.linspace(0.5, 1.5, 101):
            state = np.array([step / env.steps, 1.0 - step / env.steps, m], dtype=np.float32)
            row.append(greedy_action(online, state))
        grid.append(row)
    grid = np.array(grid)

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
    fig.savefig(os.path.join(FIGURES_DIR, "exercise_region.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
