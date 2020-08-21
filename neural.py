import numpy as np

from decor import decorFeedForward


class Neural:
    def __init__(self, countInput):
        self._weights = (2 * np.random.sample((countInput, 1))) - 1
        self._func = self.sigmoid

    @decorFeedForward
    def feedForward(self, inputs):
        total = np.dot(inputs, self._weights)
        total.shape = (1, )
        return self._func(total[0])

    def functionActivation(self):
        return self._func

    def setFunctionActivation(self, func):
        if type(func) == type(self._func):
            self._func = func

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def linear(x):
        return x

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        if w.shape == self._weights.shape:
            self._weights = w
