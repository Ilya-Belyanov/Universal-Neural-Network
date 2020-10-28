import json

from layers import Layers
from src.decor.decorNeuralNetwork import *


class NeuralNetwork:
    def __init__(self, file: str = None):
        self.layers = Layers()
        self._loss = 0

        if file is not None:
            self.load(file)

    @decorInput
    def calculate(self, inputs):
        return np.array([self.feedForward(inputs[i]) for i in range(inputs.shape[0])])

    def feedForward(self, inputs, layer=1):
        result = np.array([neural.feedForward(inputs) for neural in self.layers[layer]]).T
        if layer == len(self.layers) - 1:
            return result
        return self.feedForward(result, layer + 1)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, ls):
        self._loss = float('{:.6f}'.format(ls))

    @decorCreateStructure
    def createStructure(self, structure: list):
        self.layers.clear()
        self.layers.addLayers(len(structure) - 2)

        for i in range(len(structure)):
            count = structure[i] - 1
            if count > 0:
                self.layers.addNeural(i, count)
        return True

    def copy(self):
        NN = NeuralNetwork()
        NN.layers = self.layers.copy()
        return NN

    @decorSave
    def save(self, file: str = None):
        with open(file, 'w') as f:
            json.dump(self.weights(), f)
        return True

    @decorSave
    @decorLoad
    def load(self, file: str):
        with open(file) as f:
            weights = json.load(f)

        self.createStructure([len(layer) for layer in weights])

        for l in range(len(self.layers)):
            for i in range(self.layers.lenLayer(l)):
                self.layers.neural(l, i).weights = np.array(weights[l][i])

        return True

    def structure(self):
        return [len(layer) for layer in self.layers]

    def weights(self):
        return [[[list(w) for w in neural.weights] for neural in layer] for layer in self.layers]

    def inputs(self):
        return len(self.layers[0])

    def outputs(self):
        return len(self.layers[-1])
