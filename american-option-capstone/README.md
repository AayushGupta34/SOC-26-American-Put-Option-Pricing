# American Option Pricing Capstone

This is the final project pulling together everything from the earlier weeks: a CRR binomial tree, a neural network that learns to approximate it, and a DQN agent that learns when to exercise. The three pieces get compared against each other on the same set of contracts instead of living in separate folders that never talk to each other.

I restructured the whole thing into a proper `src/` layout for this final version rather than keeping the "one folder per week" style from before — it made the imports between pricing, the NN, and the RL side much less awkward, and it's closer to how I'd actually organize a project like this if I were starting it from scratch.

## What's where

- **`src/pricing/`** — the binomial tree (`binomial.py`), a closed-form Black-Scholes European price for a sanity cross-check (`black_scholes.py`), and the payoff functions everything else shares (`payoffs.py`).
- **`src/data/synthetic_contracts.py`** — the `OptionContract` object everything passes around, random contract sampling for training the NN, and the fixed 168-contract grid every comparison runs against.
- **`src/ml/`** — the neural network pricer: architecture (`models.py`), training (`train_nn.py`), evaluation (`evaluate_nn.py`).
- **`src/rl/`** — the exercise-policy side: the environment (`env.py`), DQN training (`train_dqn.py`), and policy evaluation (`evaluate_policy.py`).
- **`src/evaluation/`** — metrics, plots, and `comparison.py`, which is the script that actually ties the three methods together.
- **`tests/`** — binomial sanity checks, payoff checks, and RL environment terminal-behavior checks.
- **`notebooks/exploration.ipynb`** — a quick informal look at one contract before running the full grid.
- **`reports/`** — the final PDF, the figures, and a `results/` folder with the trained checkpoints and comparison CSVs so a mentor can see the numbers without rerunning anything.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` step is what makes `from pricing.binomial import crr_price` etc. work from anywhere in the repo — it's a plain setuptools src-layout package (see `pyproject.toml`). If you'd rather not install it, `conftest.py` at the root adds `src/` to the path automatically for pytest, but running the scripts directly (`python -m ml.train_nn`) still expects the install.

## Running it

The NN checkpoint and the DQN checkpoint are already committed in `reports/results/`, so you don't have to retrain anything just to look at results:

```bash
python -m evaluation.comparison   # the main comparison, writes to reports/results/
python -m pytest tests/ -v
```

If you do want to retrain from scratch:

```bash
rm reports/results/option_data.npz reports/results/neural_pricer.pt
python -m ml.train_nn        # ~a minute or two, 300 epochs

rm reports/results/dqn_online.pt
python -m rl.train_dqn       # ~a couple minutes, 10,000 episodes
```

Both are seeded (`seed=42`), so a clean rerun should land close to what's already in `reports/results/`, though the DQN's exact value will drift a little run to run since evaluation is Monte Carlo.

## The one thing worth flagging up front

The DQN was only ever trained on one contract — S0=100, K=100, T=1, r=5%, sigma=25%, 50 steps. It's not a contract-conditioned agent, so `evaluate_policy.py` does two separate checks: how it does on the contract it actually saw during training, and — as an honest out-of-distribution test — how the same trained policy performs when I run it against all 168 contracts in the shared grid without retraining anything. It does noticeably worse on the second one (it beats a naive always-hold baseline on the training contract but only on about a fifth of the full grid). That's real, not a bug, and it's the main thing the report's limitations section talks about.

## Report

`reports/Week9_AayushGupta.pdf` has the full write-up — methods, hyperparameters, results, and the limitations discussion. The source is `reports/Week9_AayushGupta.md` if you want to see it in plain text or tweak it.
