# Week 6 — Neural Network American Put Pricer

Trains a PyTorch MLP to approximate the price of an American put option
(priced via a 500-step CRR binomial tree) and evaluates how well the
network reproduces the binomial "ground truth."

All code below is exactly as provided — nothing has been added or edited.

## Files

| File | Purpose |
|---|---|
| `american_put.py` | Week 4 CRR binomial pricer (`crr_put_price`). Used to generate labels and as ground truth for evaluation. |
| `PartA.py` | Generates 10,000 synthetic contracts, labels them with `crr_put_price(..., steps=500, american=True)`, runs sanity checks, saves `week6_option_data.npz`. |
| `PartB.py` | Loads the dataset, splits 80/10/10, standardizes using training-set stats only, trains a 2-hidden-layer MLP, saves the best-validation checkpoint to `week6_neural_pricer.pt`, and saves `week6_learning_curve.png`. |
| `PartC.py` | Loads the trained model, evaluates on the test split (MAE/RMSE/max error, moneyness buckets), saves `week6_pred_vs_binomial.png` and `week6_surface_error.png`, and runs financial sanity checks. |
| `week6_learning_curve.png` | Train vs. validation MSE per epoch. |
| `week6_pred_vs_binomial.png` | Test-set scatter: NN prediction vs. binomial price. |
| `week6_surface_error.png` | Absolute error heatmap, NN surface vs. binomial surface (K=100, r=5%, sigma=25%). |
| `week6_neural_pricer.pt` | **Add your own copy** — see note below. |
| `report.md` | Short report: metrics, sanity-check results, reflection. |

## Where the two "include this" checklist items already live

**"Training script or notebook with clear feature order"** — `PartB.py`,
lines 158–166, inside the saved `artifact` dict:

```python
artifact = {
    "model_state": model.state_dict(),
    "x_mean": x_mean,
    "x_std": x_std,
    "feature_order": [
        "S0", "K", "T", "r", "sigma"
    ],
    "label_steps": 500
}
torch.save(artifact, "week6_neural_pricer.pt")
```

The feature order is stated explicitly and saved alongside the model, so
anyone loading `week6_neural_pricer.pt` later knows column 0 is `S0`,
column 1 is `K`, etc.

**"If you include a saved model, include scaling stats and architecture
code too"** — both are already present:

- **Scaling stats:** `x_mean` and `x_std` (`PartB.py` lines 36–37) are
  saved into that same `artifact` dict, so the standardization used at
  training time travels with the checkpoint.
- **Architecture code:** the `PutPricerNet` class (`PartB.py` line 65,
  re-declared identically in `PartC.py`) is included in the submitted
  scripts. `torch.save` only stores weights (`state_dict()`), not the
  class definition, so the architecture code must ship alongside the
  `.pt` file for it to be loadable — which it does here, since `PartB.py`
  and `PartC.py` are both part of the submission.

The one thing that is **not** in this folder is the actual binary
`week6_neural_pricer.pt` checkpoint itself (see note below) — the code
that produces it, the feature order, and the scaling stats are all
already in place.

## Dependencies

```
python >= 3.9
numpy
torch
matplotlib
```

```bash
pip install numpy torch matplotlib
```

## How to run from a clean session

Run in order — each part depends on a file the previous one saves:

```bash
python3 PartA.py   # -> week6_option_data.npz
python3 PartB.py   # -> week6_neural_pricer.pt, week6_learning_curve.png
python3 PartC.py   # -> prints metrics; week6_pred_vs_binomial.png, week6_surface_error.png
```

`PartA.py` loads `week6_option_data.npz` instead of regenerating it if
the file already exists, so re-running it is always safe. Contract
sampling, the train/val/test split, and PyTorch's weight init /
data-loader shuffling are all seeded (`seed=42` / `torch.manual_seed(42)`),
so a clean-session re-run reproduces the same dataset and a very similar
trained model.

## Note on `week6_neural_pricer.pt`

Not included in this folder — it's a binary checkpoint that only exists
after you actually run `python3 PartB.py` locally. Copy your own
`week6_neural_pricer.pt` into this folder before zipping up your final
submission, since `PartC.py` (and your grader, if they don't retrain)
needs it to run.
