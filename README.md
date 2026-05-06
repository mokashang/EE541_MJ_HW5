# MNIST Binary Logistic Regression (from scratch)

A NumPy-only implementation of logistic regression that learns to detect the
digit **"2"** against all other MNIST digits. The model, gradient descent
loop, regularization (none, L2, L1), and evaluation metrics are all written
without any ML framework — the notebook reads as the spec for what those
frameworks do under the hood.

## What's inside

- **Custom data pipeline** — MNIST loaded from HDF5, recast as a 1-vs-all
  binary task ("2" → 1, otherwise → 0).
- **Numerically stable sigmoid + log-loss** — uses the
  `log(1 + exp(-|z|))` reformulation to avoid overflow.
- **Training** — full-batch gradient descent with three runs:
  unregularized, L2, and L1.
- **Learning-rate selection** — sweep over `{0.01, 0.1, 0.5, 1.0}` with
  loss curves to motivate the chosen LR.
- **Diagnostics** — train/validation learning curves, 28×28 weight maps
  (which highlight the strokes the model relies on), and held-out
  accuracy / precision / recall / F1.

## Layout

```
notebook/
  binary_logistic_regression.ipynb   # full implementation + analysis
  weights.h5                         # trained weights (w: 784-vec, b: scalar)
  requirements.txt                   # numpy, h5py, matplotlib, jupyter
```

The MNIST HDF5 file (`mnist_data.h5`) is not committed — point the loader
at any standard MNIST HDF5 dump, or adapt the first cells to read the
binary IDX files instead.

## Running

```bash
pip install -r notebook/requirements.txt
jupyter notebook notebook/binary_logistic_regression.ipynb
```

## Headline numbers (test set, 10 000 samples)

| Variant | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| No regularization | 0.9810 | 0.9626 | 0.8488 | 0.9022 |
| L2 | 0.9782 | 0.9678 | 0.8159 | 0.8854 |
| L1 | 0.9756 | 0.9690 | 0.7888 | 0.8697 |

The class is imbalanced (≈10% positive), so accuracy alone is generous —
F1 is the more honest metric. The 28×28 weight visualization shows the
model relying on the curved upper stroke of "2" with negative activation
in empty corners, the kind of pattern a single linear boundary can
recover.
