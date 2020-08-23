import numpy as np
import random

from neural import Neural
from decor import decorFeedForward, decorCreateStructure


class NeuralNetwork:
    def __init__(self):
        self._layers = [[]]
        self.setClearNetwork()
        self._loss = 0

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, ls):
        self._loss = float('{:.5f}'.format(ls))

    def setClearNetwork(self):
        _layerOne = [Neural(1)]
        _layerLast = [Neural(1)]
        self._layers = [_layerOne, _layerLast]

    @decorCreateStructure
    def createStructure(self, structure: list):
        self.setClearNetwork()
        self.addLayer(len(structure) - 2)

        for i in range(len(structure)):
            count = structure[i] - 1
            if count > 0:
                self.addNeuralTo(i, count)

        return True

    def addLayer(self, count: int = 1):
        if count < 0:
            return False

        for i in range(count):
            index = len(self._layers) - 1
            self._layers.insert(index, [])
            self.addNeuralTo(index)
        self._changeInput(self.layers() - 1)
        return True

    def addNeuralTo(self, index: int, count: int = 1):
        if index < 0 or index >= len(self._layers) or count < 0:
            return False

        try:
            _inputs = len(self._layers[index - 1])
        except IndexError:
            _inputs = 1

        for i in range(count):
            self._layers[index].append(Neural(_inputs))
        self._changeInput(index + 1)

        return True

    def addInputNeural(self):
        self._layers[0].append(Neural(1))
        self._changeInput(1)

    def addOutputNeural(self):
        _inputs = len(self._layers[-2])
        self._layers[-1].append(Neural(_inputs))

    def _changeInput(self, layer):
        if layer <= 0 or layer >= len(self._layers):
            return False

        _inputs = len(self._layers[layer - 1])
        for i in range(len(self._layers[layer])):
            self._layers[layer][i] = Neural(_inputs)

    @decorFeedForward
    def calculate(self, inputs):
        if inputs.shape[1] == len(self._layers[0]):
            return [self.feedForward(inputs[i]) for i in range(inputs.shape[0])]

    def feedForward(self, inputs, layer=1):
        result = np.array([neural.feedForward(inputs) for neural in self._layers[layer]]).T
        if layer == len(self._layers) - 1:
            return result
        return self.feedForward(result, layer + 1)

    def copy(self):
        NN = NeuralNetwork()
        NN.createStructure(self.structure())
        for l in range(self.layers()):
            for i in range(self.lenLayer(l)):
                NN.layer(l)[i] = self.layer(l)[i].copy()
        return NN

    def mutation(self, prob=0.5, k=0.1):
        for l in range(1, self.layers()):
            for i in range(self.lenLayer(l)):
                weights = np.copy(self.layer(l)[i].weights)
                mutantW = np.zeros(weights.shape)
                for w in range(weights.shape[0]):
                    if random.random() < prob:
                        mutantW[w][0] = k * (2 * random.random() - 1)
                self.layer(l)[i].weights = np.copy(weights + mutantW)

        return self

    def structure(self):
        return [len(layer) for layer in self._layers]

    def weights(self):
        return [[neural.weights for neural in layer] for layer in self._layers]

    def layers(self):
        return len(self._layers)

    def lenLayer(self, index: int):
        if index < 0 or index >= self.layers():
            return False
        return len(self._layers[index])

    def layer(self, index: int):
        if index < 0 or index >= self.layers():
            return False
        return self._layers[index]
