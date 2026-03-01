# EE 541 - Homework 5

**Mengjia Shang** | USC ID: 7338151449

## Problem 1: Logistic Regression Binary Classifier

Implementation of a logistic regression classifier from scratch to detect the digit "2" in the MNIST dataset.

### Repository Structure

```
README.md
q1/
  q1.ipynb         # Jupyter notebook with code and analysis
  q1.pdf           # Writeup with answers to all questions
  weights.h5       # Trained model weights (w: 784-vector, b: scalar)
  mnist_data.h5    # MNIST dataset in HDF5 format
  requirements.txt # Python dependencies
```

### Setup

```bash
pip install -r q1/requirements.txt
```

### Running

Open `q1/q1.ipynb` in Jupyter and run all cells. The notebook will:
1. Load and preprocess the MNIST data
2. Train logistic regression models with no regularization, L2, and L1
3. Generate plots comparing learning rates, learning curves, and weight visualizations
4. Evaluate and report test metrics (accuracy, precision, recall, F1)
5. Save the trained weights to `weights.h5`
