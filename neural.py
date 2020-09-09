import numpy as np

from src.decor.decorNeuralNetwork import decorInput
from src.functions import sigmoid, linear


class Neural:
    def __init__(self, countInput):
        self._weights = (2 * np.random.sample((countInput, 1))) - 1
        self._bias = 0
        self._func = sigmoid

    @decorInput
    def feedForward(self, inputs):
        total = np.dot(inputs, self._weights)
        total.shape = (1,)
        return self._func(total[0] + self._bias)

    def functionActivation(self):
        return self._func

    def setFunctionActivation(self, func):
        if type(func) == type(self._func):
            self._func = func

    def copy(self):
        n = Neural(self.weights.shape[0])
        n.setFunctionActivation(self._func)
        n.weights = np.copy(self.weights)
        return n

    def inputs(self):
        return self._weights.shape[0]

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights.shape == self._weights.shape:
            self._weights = weights

    def weight(self, index: int):
        if index < 0 or index >= self._weights.shape[0]:
            return None
        return self._weights[index][0]

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        if isinstance(bias, (int, float)):
            self._bias = bias
