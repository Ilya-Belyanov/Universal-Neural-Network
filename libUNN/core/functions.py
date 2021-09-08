import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def border(x):
    return 1 if x >= 0.5 else 0


def linear(x):
    return x
