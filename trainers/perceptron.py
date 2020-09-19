import numpy as np

from neuralNetwork import NeuralNetwork


class Perceptron:
    def learn(self, neuralNetwork: NeuralNetwork, trainingInput, trainingOutput, count: int = 100):
        if neuralNetwork.structure() != [3, 1]:
            return None
        for _ in range(count):
            output = neuralNetwork.calculate(trainingInput)
            err = trainingOutput - output
            adjustments = np.dot(trainingInput.T, err * (output * (1 - output)))
            neuralNetwork.neural(1, 0).weights = neuralNetwork.neural(1, 0).weights + adjustments

        return neuralNetwork