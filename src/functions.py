import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def border(x):
    if x >= 0.5:
        return 1
    return 0


def linear(x):
    return x
