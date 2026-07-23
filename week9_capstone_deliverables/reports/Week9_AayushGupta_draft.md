# Week 9 Final Capstone — American Put Option Pricing
### CRR Binomial vs. Neural-Network Pricer vs. RL Exercise Policy

*Draft — written in the third person from computed results. Convert to PDF
and rewrite the reflection/discussion sections in your own voice before
submitting; the numbers themselves come straight from `compare_all.py`'s
output and don't need to change unless you rerun with different seeds/grids.*

---

## 1. Executive Summary

This capstone integrates three separate weekly assignments — a CRR binomial
pricer (Week 4), a neural-network price approximator (Week 6), and a
DQN-based early-exercise policy (Weeks 7–8) — into one evaluation. The NN
pricer was compared against the binomial benchmark across a shared grid of
168 contracts spanning deep in/out-of-the-money puts, short and long
maturities, and low and high volatility, achieving a mean absolute error of
**0.083** (RMSE 0.103) against binomial prices ranging roughly from under $1
to $60. The RL policy was validated against the binomial-optimal exercise
boundary for the single contract it was trained on: its exercise decisions
agreed with the binomial-optimal decision **62.6%** of the time, and its
realized value (6.21) came in below both the `always-hold-to-expiry`
baseline (7.45) and the binomial benchmark (7.95) — an honest result
discussed in Section 9.

## 2. Problem Statement

An American put gives the holder the right to sell the underlying at strike
`K` at any time up to expiry `T`. Pricing it requires solving an optimal
stopping problem: unlike a European put, there is no closed-form solution,
because the holder's optimal exercise decision depends on the entire future
path of the stock. This capstone compares three ways of approaching that
problem — an exact numerical method (binomial tree), a learned function
approximator for the price (neural network), and a learned decision policy
for the exercise timing itself (reinforcement learning) — and asks how well
the latter two approximate the first.

## 3. Financial Background

A put payoff at any exercise time is `max(K - S, 0)`. American-style options
are worth at least as much as their European counterparts, since the holder
has strictly more choices (exercise whenever vs. only at expiry); the
difference is the *early-exercise premium*. For a put, early exercise
becomes attractive when the option is deep in-the-money and there isn't much
time value left to lose — which is exactly the "exercise boundary" shape
this report traces out in Section 8.

## 4. The Binomial Method (Week 4)

`week4_AayushGupta/american_put.py::crr_put_price` builds a Cox-Ross-Rubinstein
recombining binomial tree with up/down factors `u = exp(sigma*sqrt(dt))`,
`d = 1/u`, and risk-neutral probability `p = (exp(r*dt) - d)/(u - d)`, then
rolls the option value back from expiry, taking the max of continuation
value and immediate exercise value at every node for the American case. This
is treated as ground truth throughout the rest of the capstone: both the NN
labels (Week 6) and the RL environment's dynamics (Weeks 7–8) are built from
this same risk-neutral model.

## 5. The Neural-Network Method (Week 6)

A 2-hidden-layer MLP (`PutPricerNet`, 128 units/layer) was trained on 10,000
contracts sampled uniformly over `S0 ∈ [60,140]`, `K ∈ [80,120]`,
`T ∈ [0.05, 2.0]`, `r ∈ [0, 0.10]`, `sigma ∈ [0.10, 0.50]`, labeled with the
500-step binomial price. Inputs are standardized using training-set
mean/std only (no leakage from val/test). Training ran 300 epochs with
Adam, keeping the best-validation-MSE checkpoint. Best validation MSE:
**0.0105**.

## 6. The RL Method (Weeks 7–8)

Week 7 first framed early exercise as an MDP with a 2-feature state
`[time_fraction, moneyness]` and a small tabular Q-learning agent (20×30
state bins). Week 8 is the version actually used going forward: a 3-feature
state `[time_fraction, time_to_expiry, moneyness]`, a full DQN (2-hidden-layer
MLP, target network, replay buffer, ε-greedy exploration decaying 1.0→0.05
over 10,000 episodes), trained against the exact same risk-neutral binomial
dynamics as Week 4's tree for one fixed contract: `S0=100, K=100, T=1.0,
r=0.05, sigma=0.25, steps=50`.

## 7. Experimental Setup

- **Shared grid** (`week9_AayushGupta/contract_grid.py`): `S0 ∈ {70,...,130}`
  (step 10), `K=100`, `T ∈ {0.25, 0.5, 1.0, 2.0}`, `r ∈ {0.02, 0.05}`,
  `sigma ∈ {0.15, 0.25, 0.40}`, `steps=100` → **168 contracts**.
- **NN vs. binomial**: every contract in the grid, priced both ways.
- **RL vs. binomial**: single contract (`S0=100, K=100, T=1.0, r=0.05,
  sigma=0.25, steps=50` — the one `dqn_online.pt` was trained on), evaluated
  over 10,000 fresh episodes per policy for the value/exercise-rate table,
  and over a 50-step × 81-moneyness-point grid for boundary agreement.
- **Seeds**: Week 6 training used `seed=42` throughout (data sampling,
  train/val/test split, weight init). Week 8's DQN and evaluation both used
  `seed=42`. Policy evaluation in this report uses fresh Monte Carlo seeds
  (0..9999) independent of training.
- **Metrics**: MAE, RMSE, max absolute error, mean bias, median absolute
  error, mean relative error (pricing); Monte Carlo value + standard error,
  exercise rate, average exercise step (RL policy); boundary agreement
  fraction (RL vs. binomial-optimal decision).

## 8. Results

### 8.1 Pricing accuracy — NN vs. binomial (full grid, n=168)

| Metric | Value |
|---|---|
| MAE | 0.0832 |
| RMSE | 0.1035 |
| Max absolute error | 0.2829 |
| Mean bias | -0.0132 |
| Median absolute error | 0.0706 |
| Mean relative error | 1.334 |
| Intrinsic-value violations | 24 / 168 |
| Spot-monotonicity violations (K=100,T=1,r=5%,sigma=25% slice) | 0 |

The mean relative error of 1.33 looks alarming in isolation but is an
artifact of contracts whose true binomial price is very close to zero
(deep OTM, short maturity) — a tiny absolute error becomes a huge relative
one when the denominator is near zero. MAE and RMSE are the more meaningful
headline numbers here; both are small relative to the price range in the
grid (roughly $0–$60).

**By moneyness:**

| Bucket | n | NN MAE | NN RMSE | NN bias | Binomial mean | NN mean |
|---|---|---|---|---|---|---|
| ITM put | 72 | 0.0910 | 0.1116 | -0.0063 | 21.97 | 21.96 |
| ATM | 48 | 0.1044 | 0.1229 | -0.0317 | 6.66 | 6.63 |
| OTM put | 48 | 0.0503 | 0.0608 | -0.0050 | 2.81 | 2.80 |

**By maturity:**

| Bucket | n | NN MAE | NN RMSE | NN bias |
|---|---|---|---|---|
| Long (T>0.5y) | 84 | 0.0733 | 0.0931 | +0.0006 |
| Short (T≤0.5y) | 84 | 0.0931 | 0.1129 | -0.0270 |

**By volatility:**

| Bucket | n | NN MAE | NN RMSE | NN bias |
|---|---|---|---|---|
| High vol (σ>0.20) | 112 | 0.0853 | 0.1023 | +0.0050 |
| Low vol (σ≤0.20) | 56 | 0.0791 | 0.1058 | -0.0495 |

The network is most accurate for deep OTM puts and slightly worse near the
money, and slightly worse for short-dated, low-volatility contracts — likely
because these have smaller absolute price levels and sharper curvature near
expiry, both harder for a fixed-capacity MLP to fit precisely.

24 of 168 contracts (14%) had NN predictions fall (very slightly) below
intrinsic value — expected, since the network minimizes squared price error
rather than being explicitly constrained to respect the intrinsic-value
floor.

### 8.2 RL policy comparison (single contract, 10,000 evaluation episodes)

| Policy | Value | Std. error | Exercise rate | Avg. exercise step |
|---|---|---|---|---|
| always-hold-to-expiry | 7.4479 | 0.1098 | 0.000 | — |
| **dqn** | **6.2106** | 0.0435 | 0.787 | 18.6 |
| random | 0.9742 | 0.0222 | 1.000 | 1.0 |
| immediate-exercise | 0.0000 | 0.0000 | 1.000 | 0.0 |
| **binomial (American)** | **7.9520** | — | — | — |

### 8.3 Boundary agreement

The DQN's hold/exercise decision agreed with the binomial-optimal decision
implied by the exercise boundary on **62.6%** of a 50-step × 81-moneyness
grid of states.

## 9. Discussion: Strengths, Failures, and Limitations

**NN pricer:** strong overall accuracy (MAE ~0.08 against prices up to ~$60)
with no monotonicity violations and only mild, small intrinsic-value
violations. Weakest near the money and for short-dated contracts.

**RL policy — the honest result:** the DQN's value (6.21) is *below* both
the binomial optimum (7.95) and the naive always-hold baseline (7.45), and
its decisions only agree with the true optimal boundary 62.6% of the time.
This means Week 8's DQN, despite being a full replay-buffer + target-network
setup (a meaningful step up from Week 7's small tabular Q-learner), has not
fully converged to the optimal exercise policy for its one training
contract. This is consistent with Week 7's own finding that a much simpler
Q-learning agent also underperformed always-hold — the takeaway carries
forward rather than being resolved by moving to deep RL. Plausible causes:
10,000 training episodes with a single terminal reward per episode is a
sparse-reward setting; more episodes, reward shaping, or a lower final
epsilon might close the gap.

**Scope limitation — RL was validated on one contract, not the grid:** unlike
the NN pricer, `dqn_online.pt` was trained against one specific contract's
risk-neutral dynamics. Nothing in this report claims the trained policy
generalizes to other strikes, maturities, or volatilities; extending it
would require training a contract-conditioned RL agent (e.g. adding
`S0, K, T, r, sigma` to the state), which is future work, not something this
capstone attempts.

**Modeling simplifications common to all three methods:** discrete
time steps rather than continuous exercise opportunity; no dividends; no
transaction costs; constant, known volatility (no smile/surface, no
stochastic volatility); a single, fixed risk-free rate over the option's
life. The NN's labels come directly from the binomial tree, so the network
can only be as correct as its teacher — it cannot outperform the binomial
model it was trained to imitate, only approximate it faster. The RL policy's
evaluation is Monte Carlo, so the reported values carry real standard error
(shown above) rather than being exact.

## 10. Reproducibility

Full instructions are in the repository root README. In short:

```bash
pip install -r requirements.txt
cd week6 && python3 PartA.py && python3 PartB.py && cd ..
cd week9_AayushGupta && python3 compare_all.py && python3 make_figures.py
python3 -m pytest tests/ -v
```

`week8_AayushGupta/dqn_online.pt` is already committed, so retraining the
DQN (`python3 week8_AayushGupta/week8.py`) is optional and only needed to
reproduce it from scratch. All seeds are fixed (`seed=42` throughout Weeks
6 and 8); Week 9's Monte Carlo policy evaluation uses fresh seeds
independent of training, so its exact value/std-error numbers will vary
slightly run to run within the reported standard error.

## 11. Conclusion

Bringing the binomial tree, the neural-network pricer, and the RL exercise
policy together onto one shared, honestly-reported evaluation shows a mixed
but genuinely informative picture: the NN pricer is a strong, fast
approximation to the binomial benchmark across a wide range of contracts,
while the RL policy — though structurally capable of learning a
state-dependent exercise rule, unlike the fixed baselines — has not yet
converged to the true optimal policy for even the single contract it was
trained on. That gap, and the fact that the RL result doesn't extend beyond
one contract, are the two most important limitations for anyone building on
this work next.

---

**Repository:** `AayushGupta34/SOC-26-American-Put-Option-Pricing`
**Figures:** `reports/figures/` (new: payoff diagram, standalone exercise
boundary, full-grid NN-vs-binomial scatter) plus `week6/week6_learning_curve.png`,
`week6/week6_pred_vs_binomial.png`, `week6/week6_surface_error.png`,
`week8_AayushGupta/exercise_region.png`.
