import numpy as np

from src.odj.neuralNetwork import NeuralNetwork
from src.decor.decorBreeder import decorTraining


class Perceptron2L:

    @decorTraining
    def learn(self, neuralNetwork: NeuralNetwork, trainingInput, trainingOutput, count: int = 100):
        if len(neuralNetwork.layers) != 2 or neuralNetwork.layers.lenLayer(1) != 1:
            return None
        for _ in range(count):
            output = neuralNetwork.calculate(trainingInput)
            err = trainingOutput - output
            adjustments = np.dot(trainingInput.T, err * (output * (1 - output)))
            neuralNetwork.layers.neural(1, 0).weights = neuralNetwork.layers.neural(1, 0).weights + adjustments

        return neuralNetwork
