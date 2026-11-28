# Snake: SARSA vs Q-learning — submission

Final agent: **SARSA, fully observable** board (10x10), cosine learning-rate
decay 1e-3 -> 1e-4, corrected self-collision penalty, action masks for walls
and reversals, per-episode reset. Seed 0.

## Files

| File | Content |
|---|---|
| `report.pdf` | the report |
| `environments_fully_observable_2.py`, `environments_partially_observable_2.py` | environments used (fully and partially observable, 5x5 window) |
| `baseline.py` | heuristic baseline (no learning), evaluated in both environments |
| `train.py` | training code of the final agent (SARSA); `OSSERVABILITA`, `SHAPING`, `LR_DECAY` select the variants discussed in the report |
| `evaluate.py` | loads `weights/q_network.keras` and evaluates the greedy agent against the baseline |
| `weights/q_network.keras` | weights of the final trained agent |

## Run

```
python evaluate.py     # evaluation of the trained agent (about 15 s on CPU)
python baseline.py     # heuristic baseline only
python train.py        # retrains the agent from scratch (about 1.5 h on CPU)
```

Requirements: Python 3.10, TensorFlow 2.x, NumPy, Matplotlib (Pillow, a
Matplotlib dependency, is used to save GIFs during training).
Tested with Python 3.10, TensorFlow 2.21, NumPy 2.2, Matplotlib 3.10.

The numbers in the report come from training runs without a fixed seed; the
delivered weights come from a rerun of the same configuration with seed 0.
