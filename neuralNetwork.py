import json

from neural import Neural
from src.decor.decorNeuralNetwork import *


class NeuralNetwork:
    def __init__(self, file: str = None):
        self._layers = [[]]
        self.clearNetwork()
        self._loss = 0

        if file is not None:
            self.load(file)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, ls):
        self._loss = float('{:.6f}'.format(ls))

    def clearNetwork(self):
        _layerOne = [Neural(1)]
        _layerLast = [Neural(1)]
        self._layers = [_layerOne, _layerLast]

    @decorCreateStructure
    def createStructure(self, structure: list):
        self.clearNetwork()
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

    @decorRemoveLayer
    def removeLayer(self, start: int, end: int = None):
        for l in range(end - start):
            self._layers.pop(start)
        self._changeInput(start)
        return True

    def addNeuralTo(self, index: int, count: int = 1):
        if index < 0 or index >= self.layers() or count < 0:
            return False

        try:
            _inputs = len(self._layers[index - 1])
        except IndexError:
            _inputs = 1

        for i in range(count):
            self._layers[index].append(Neural(_inputs))
        self._changeInput(index + 1)
        return True

    @decorNeural
    def deleteNeuralFrom(self, layer, index):
        self._layers[layer].pop(index)
        if self.lenLayer(layer) == 0:
            self.removeLayer(layer)
        else:
            self._changeInput(layer + 1)
        return True

    def addInputNeural(self):
        self._layers[0].append(Neural(1))
        self._changeInput(1)

    def addOutputNeural(self):
        _inputs = len(self._layers[-2])
        self._layers[-1].append(Neural(_inputs))

    @decorLayer
    def _changeInput(self, layer):
        _inputs = len(self._layers[layer - 1])
        for i in range(len(self._layers[layer])):
            self._layers[layer][i] = Neural(_inputs)
        return True

    @decorInput
    def calculate(self, inputs):
        return np.array([self.feedForward(inputs[i]) for i in range(inputs.shape[0])])

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

        for l in range(self.layers()):
            for i in range(self.lenLayer(l)):
                self.neural(l, i).weights = np.array(weights[l][i])

        return True

    def structure(self):
        return [len(layer) for layer in self._layers]

    def weights(self):
        return [[[list(w) for w in neural.weights] for neural in layer] for layer in self._layers]

    def layers(self):
        return len(self._layers)

    @decorLayer
    def lenLayer(self, layer: int):
        return len(self._layers[layer])

    @decorLayer
    def layer(self, layer: int):
        return self._layers[layer]

    @decorNeural
    def neural(self, layer: int, index: int):
        if index < 0 or index >= self.lenLayer(layer):
            return False
        return self._layers[layer][index]

    def inputs(self):
        return len(self._layers[0])

    def outputs(self):
        return len(self._layers[-1])
