from ..neuralNetwork.neural import Neural
from ..decor.decorLayers import *


class LayersIterator:
    def __init__(self, layers):
        self.layers = layers
        self.el = 0

    def __next__(self):
        if self.el < len(self.layers):
            self.el += 1
            return self.layers[self.el - 1]
        else:
            raise StopIteration


class Layers:
    def __init__(self):
        self._layers = list()
        self.clear()

    def __len__(self):
        return len(self._layers)

    @decorLenLayer
    def lenLayer(self, layer: int):
        return len(self._layers[layer])

    def __str__(self):
        return "Layers with {} layer".format(len(self._layers))

    def __iter__(self):
        return LayersIterator(self)

    def __getitem__(self, key):
        return self._layers[key]

    def append(self, layer):
        self._layers.append(layer)

    def neural(self, layer: int, index: int):
        if index < 0 or index >= len(self._layers[layer]):
            return False
        return self._layers[layer][index]

    def addLayers(self, count: int = 1):
        if count < 0:
            return False

        for i in range(count):
            index = len(self._layers) - 1
            self._layers.insert(index, [])
            self.addNeural(index)

        self._changeInput(len(self._layers) - 1)
        return True

    def removeLayer(self, pos: int):
        if pos < 0 or pos >= len(self._layers):
            return False
        self._layers.pop(pos)
        self._changeInput(pos)
        return True

    @decorAddNeuralTo
    def addNeural(self, layer: int, count: int = 1):
        try:
            _inputs = len(self._layers[layer - 1])
        except IndexError:
            _inputs = 1

        for i in range(count):
            self._layers[layer].append(Neural(_inputs))
        self._changeInput(layer + 1)
        return True

    def addInputNeural(self):
        self._layers[0].append(Neural(1))
        self._changeInput(1)

    def addOutputNeural(self):
        _inputs = len(self._layers[-2])
        self._layers[-1].append(Neural(_inputs))

    @decorDeleteNeuralFrom
    def deleteNeural(self, layer, index):
        self._layers[layer].pop(index)
        if len(self._layers[layer]) == 0:
            self.removeLayer(layer)
        else:
            self._changeInput(layer + 1)
        return True

    def clear(self):
        self._layers = [[Neural(1)], [Neural(1)]]

    def clearForce(self):
        self._layers = list()

    def _changeInput(self, layer):
        if layer < len(self._layers):
            _inputs = len(self._layers[layer - 1])
            _countNeural = len(self._layers[layer])
            self._layers[layer] = [Neural(_inputs) for _ in range(_countNeural)]

    def copy(self):
        layers = Layers()
        layers.clearForce()
        for l in self._layers:
            layer = []
            for n in l:
                layer.append(n.copy())
            layers.append(layer)
        return layers
