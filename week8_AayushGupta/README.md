# Week 8 — American Put Option Pricing: RL vs. Binomial Benchmark

This week compares a **Deep Q-Network (DQN)** agent, trained to find the optimal exercise
policy for an American put, against the closed-form-numerical **Cox-Ross-Rubinstein (CRR)
binomial tree** price for the same contract.

## Files

| File | Purpose |
|---|---|
| `american_put.py` | CRR binomial tree pricer for European/American puts. Provides `crr_put_price(...)`, used both as a standalone pricer and as the benchmark for the RL agent. |
| `week8.py` | Builds a binomial-lattice environment with the *same* risk-neutral dynamics as the CRR model, trains a DQN agent to decide **hold vs. exercise** at each step, evaluates it against baseline policies and the CRR benchmark, and plots the learned exercise region. |

## Why this comparison is valid

The RL environment (`AmericanPutEnv`) is built with the identical `u`, `d`, and risk-neutral
probability `p` as the CRR tree in `american_put.py`. This means the DQN agent is solving the
*same* optimal-stopping problem, under the same risk-neutral measure, that the binomial tree
solves exactly via backward induction. So the DQN's estimated option value can be compared
directly against `crr_put_price(..., american=True)` as ground truth, and the gap between them
is a measure of how close the learned policy is to the true optimal exercise boundary.

## Model / Environment Setup

- **Contract**: `S0=100, K=100, T=1.0, r=0.05, sigma=0.25`, `steps=50`
- **State**: `[time_fraction, time_to_expiry, moneyness]` (3-dim, `moneyness = S/K`)
- **Actions**: `0 = hold`, `1 = exercise`
- **Reward**: `0` while holding, intrinsic payoff `max(K - S, 0)` when the episode ends
  (exercise or expiry)
- **Agent**: 2-hidden-layer MLP Q-network (64 units), trained with a target network,
  replay buffer, ε-greedy exploration (decaying 1.0 → 0.05), Huber (`smooth_l1`) loss,
  gradient clipping, Adam optimizer

## How to Run

```bash
pip install numpy torch matplotlib
python week8.py
```

This will:
1. Train the DQN for 10,000 episodes, logging discounted payoff, exploration rate,
   loss, and exercise rate every 500 episodes.
2. Save trained weights to `dqn_online.pt`.
3. Evaluate the DQN policy against baseline policies (`always-hold-to-expiry`,
   `immediate-exercise`, `random`) over 5,000 fresh episodes each, and print each policy's
   estimated value, standard error, and exercise statistics.
4. Print the CRR binomial benchmark price (American and European) for direct comparison.
5. Save a heatmap of the learned exercise region (`exercise_region.png`) — green regions
   indicate states where the agent chooses to exercise.

To use the pricer independently:

```python
from american_put import crr_put_price

price = crr_put_price(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=50, american=True)
```

## Output

Console output includes a comparison table:

```
policy                       value   std_err   ex_rate  avg_ex_step
always-hold-to-expiry       ...
immediate-exercise          ...
random                      ...
dqn                         ...
binomial (American)         ...
binomial (European)         ...
```

The DQN value should sit close to the American binomial benchmark, and (as expected for
a put) both should be ≥ the European binomial value, reflecting the value of early exercise.

`exercise_region.png` visualizes the learned policy as a function of time and moneyness,
with the at-the-money line marked for reference. Note: the grid sweeps moneyness over
`[0.5, 1.5]` at every time step, including combinations the binomial tree can't actually
reach that early/late — the agent's behavior there is extrapolation, not a trained region.

## Notes / Limitations

- Training uses the risk-neutral measure (matching the CRR tree) rather than a real-world
  drift, since the goal here is to validate the RL approach against a known-correct price,
  not to simulate realistic physical-measure paths.
- Results are stochastic; re-running with a different seed will shift the DQN's value
  slightly, though it should stay close to the binomial benchmark given the training length.
- `week8.py` currently executes training/evaluation/plotting at import time (no
  `if __name__ == "__main__":` guard) — fine for running as a script, but worth wrapping
  if this module is ever imported elsewhere.
