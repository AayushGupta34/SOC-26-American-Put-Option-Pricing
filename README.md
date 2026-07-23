# American Option Pricing Capstone (SOC 26)

This repository compares three approaches to pricing and exercising an American
put option:

1. **CRR binomial tree** (Week 4) — the ground-truth numerical benchmark.
2. **Neural-network pricer** (Week 6) — an MLP trained to approximate the
   binomial price directly from contract parameters.
3. **RL exercise policy** (Weeks 7–8) — a DQN agent trained to decide
   hold-vs-exercise at each step of a single contract's life.

Week 9 (`week9_AayushGupta/`) ties all three together: it runs the binomial
pricer and NN pricer across one shared grid of 168 contracts spanning deep
ITM/ATM/OTM puts, short/long maturities, and low/high volatility, and
validates the trained DQN's exercise decisions against the binomial-optimal
exercise boundary for the one contract it was trained on.

## Repository map

| Folder | Contents |
|---|---|
| `week4_AayushGupta/` | CRR binomial pricer (`american_put.py`, `crr_put_price`), convergence + price-surface + exercise-boundary report, sanity tests. |
| `week6/` | Neural-network pricer: dataset generation (`PartA.py`), training (`PartB.py`), evaluation (`PartC.py`). See note below on the checkpoint. |
| `week7_AayushGupta/` | First pass at framing early exercise as an MDP; a 2-feature `[time_fraction, moneyness]` environment and a small tabular Q-learning agent. Superseded by Week 8 for anything Week 9 reuses — see that folder's README for why. |
| `week8_AayushGupta/` | The environment and DQN actually used going forward: 3-feature state `[time_fraction, time_to_expiry, moneyness]`, trained checkpoint `dqn_online.pt`. |
| `week9_AayushGupta/` | This capstone's new integration code — see below. |
| `reports/` | Final report (`Week9_AayushGupta.pdf`, drafted as `.md` — see note) and `figures/`. |
| `Week1_AAYUSHGUPTA.pdf`, `Week2_AayushGupta.pdf`, `week3_AayushGupta.pdf`, `Week5_AayushGupta/` | Earlier weeks' submissions, unrelated to the pricing/RL pipeline. |

(Folder-naming is inconsistent across early weeks — `Week5_AayushGupta` vs.
`week6` vs. `week4_AayushGupta` — left as-is rather than renamed, since
renaming already-submitted folders would break existing links to them.)

## Week 9 integration code

Everything under `week9_AayushGupta/` imports from the folders above instead
of duplicating their logic:

| File | Purpose |
|---|---|
| `option_contract.py` | `OptionContract` dataclass — one consistent object for `S0, K, T, r, sigma, steps` used everywhere, so the NN and RL comparisons can't accidentally use different parameter orderings or grids. |
| `contract_grid.py` | The shared 168-contract evaluation grid. |
| `pricers.py` | `BinomialAmericanPutPricer`, a thin wrapper around `week4_AayushGupta/american_put.py::crr_put_price`. |
| `nn_pricer.py` | Loads `week6/week6_neural_pricer.pt` (architecture mirrors `week6/PartB.py::PutPricerNet` exactly — see the file's docstring for why it's a deliberate copy, not an import). |
| `pricing_metrics.py` | MAE/RMSE/bias + intrinsic-value and monotonicity sanity checks. |
| `boundary_utils.py` | Exercise-boundary extraction, refactored from `week4_AayushGupta/week4_report.py`'s inline plotting code into a reusable `{step: boundary_spot}` dict. |
| `rl_policy.py` | Loads `week8_AayushGupta/dqn_online.pt`, wraps `week8.py::evaluate_policy` into a comparison table, and adds a new `boundary_agreement` check (DQN decisions vs. the binomial-optimal exercise boundary). |
| `compare_all.py` | Runs everything above and writes CSVs + a summary to `week9_AayushGupta/output/`. |
| `make_figures.py` | Generates the put payoff diagram, a standalone exercise-boundary figure, and an NN-vs-binomial scatter over the full shared grid. |
| `tests/test_capstone.py` | Re-validates Week 4's binomial invariants through the new wrapper, plus a payoff test and an RL environment terminal-behavior test. |

**Important scope note:** the NN-vs-binomial comparison runs across the full
168-contract grid, but the RL comparison does not — `week8_AayushGupta`'s
environment and `dqn_online.pt` were trained against one specific contract
(S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=50), not the shared grid.
The RL section of this report validates that one trained policy, it doesn't
claim the DQN generalizes across contracts. See `reports/` for the full
discussion.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce results

Some steps depend on artifacts from earlier weeks that are regenerated
locally rather than committed as binaries (see note below). From a clean
clone, in order:

```bash
# 1. Generate the Week 6 training dataset and train the NN pricer
#    (only needed if week6/week6_neural_pricer.pt isn't already present)
cd week6
python3 PartA.py   # -> week6_option_data.npz
python3 PartB.py   # -> week6_neural_pricer.pt, week6_learning_curve.png
cd ..

# 2. (Optional) retrain the DQN -- dqn_online.pt is already committed in
#    week8_AayushGupta/, so this step can be skipped unless you want to
#    retrain from scratch (takes a few minutes, 10,000 episodes):
cd week8_AayushGupta
python3 week8.py
cd ..

# 3. Run the Week 9 comparison and figures
cd week9_AayushGupta
python3 compare_all.py
python3 make_figures.py
python3 -m pytest tests/ -v
```

## Note on `week6_neural_pricer.pt`

Not committed to the repo (binary checkpoint, only exists after running
`week6/PartB.py` locally — see `week6/README.md`). `week9_AayushGupta/nn_pricer.py`
will raise a clear `FileNotFoundError` with the regeneration command if it's
missing when you run `compare_all.py`.

## Final report

See `reports/Week9_AayushGupta_draft.md` for the full write-up (executive
summary, methods, results, limitations, reproducibility). Convert to PDF
before submitting through the course's Google Form, alongside the repo link
and final figures.
