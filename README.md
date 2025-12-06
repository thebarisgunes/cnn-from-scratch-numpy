# CNN From Scratch with NumPy

This repo contains an educational Jupyter notebook that implements a small Convolutional Neural Network **from scratch using only NumPy** and the MNIST dataset.

The goal is to **understand** how a CNN works internally, not to get state-of-the-art accuracy.

## What’s inside

- `cnn_from_scratch.ipynb` – main notebook:
  - Loads MNIST from `tensorflow.keras.datasets`
  - Implements:
    - `Conv3x3` layer
    - `ReLU` activation
    - `MaxPool2` layer
    - `Softmax` fully connected layer (with cross-entropy)
  - Trains the CNN for a few epochs on a small subset of MNIST (e.g. 1000 images)
  - Evaluates test loss and accuracy

The architecture:

```text
Input (28×28 grayscale)
 → Conv3x3 (8 filters)
 → ReLU
 → MaxPool2×2
 → Softmax (fully connected)
 → 10-class output (digits 0–9)
