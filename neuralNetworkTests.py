import unittest
import random

from neuralNetworl import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


class NeuralNetworkTest(unittest.TestCase):
    def testNetworkCalculate(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addLayer(2)
        n.addNeuralTo(1)

        self.assertEqual(type(n.calculate(np.array([0, 0]))), type([]),
                         "Should be Equal")

        self.assertEqual(type(n.calculate(np.array([[0, 1],
                                                    [1, 0]]))), type([]),
                         "Should be Equal")

    def testNetworkInput(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addLayer()

        self.assertEqual(n.calculate(np.array([0, 0])), n.calculate([[0, 0]]),
                         "Should be Equal")

        self.assertEqual(n.calculate(np.array([1, 1])), n.calculate([1, 1]),
                         "Should be Equal")

        self.assertEqual(n.calculate(np.array([[1, 1]])), n.calculate([[1, 1]]),
                         "Should be Equal")

        self.assertEqual(n.calculate(np.array([[1, 1], [0, 0]])),
                         n.calculate([[1, 1], [0, 0]]),
                         "Should be Equal")

        self.assertEqual(n.calculate(np.array([0, 0, 0])), None,
                         "Should be None")
        self.assertEqual(n.calculate([0, 0, 0]), None,
                         "Should be None")
        self.assertEqual(n.calculate(np.array([1])), None,
                         "Should be None")
        self.assertEqual(n.calculate([1]), None,
                         "Should be None")
        self.assertEqual(n.calculate(np.array([[1]])), None,
                         "Should be None")
        self.assertEqual(n.calculate([[1]]), None,
                         "Should be None")

    def testNetworkCopy(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addLayer(2)
        n.addNeuralTo(1)
        n2 = n.copy()
        wn = n.weights()
        wn2 = n2.weights()
        for layer in range(len(wn)):
            for neural in range(len(wn[layer])):
                for weight in range(len(wn[layer][neural])):
                    self.assertEqual(wn[layer][neural][weight], wn2[layer][neural][weight], "Should be Equal")

    def testNetworkMutant(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addLayer()
        n2 = n.copy()
        n.mutation()
        self.assertEqual(n.structure(), n2.structure(), "Should be Equal")

    def testNetworkChanged(self):
        n = NeuralNetwork()
        self.assertTrue(n.addLayer())
        self.assertTrue(n.addLayer())
        self.assertTrue(n.addNeuralTo(0))
        self.assertTrue(n.addNeuralTo(1))
        self.assertTrue(n.addNeuralTo(2))
        self.assertTrue(n.addNeuralTo(3))
        self.assertFalse(n.addNeuralTo(4))

    def testNetworkSetStructure(self):
        n = NeuralNetwork()
        structure = [1, 1, 1]
        n.createStructure(structure)
        self.assertEqual(n.structure(), [1, 1, 1], "Should be [1, 1, 1]")

        structure = [0, 1, 0]
        n.createStructure(structure)
        self.assertEqual(n.structure(), [1, 1, 1], "Should be [1, 1, 1]")

        structure = [0, 1, 0]
        n.createStructure(structure)
        self.assertEqual(n.structure(), [1, 1, 1], "Should be [1, 1, 1]")

        structure = [0, 0]
        n.createStructure(structure)
        self.assertEqual(n.structure(), [1, 1], "Should be [1, 1]")

        structure = [0, 4, 4, 5, 0]
        n.createStructure(structure)
        self.assertEqual(n.structure(), [1, 4, 4, 5, 1], "Should be [1, 4, 4, 5, 1]")

        structure = [0]
        self.assertFalse(n.createStructure(structure))

        structure = [0, 0, 0, 0]
        self.assertFalse(n.createStructure(structure))

        structure = [0, 4, 0, 5, 0]
        self.assertFalse(n.createStructure(structure))

        structure = [0, -1, 4, 5, 0]
        self.assertFalse(n.createStructure(structure))

    def testNetworkStructure(self):
        n = NeuralNetwork()
        self.assertEqual(n.structure(), [1, 1], "Should be Equal [1, 1]")
        n.addLayer()
        self.assertEqual(n.structure(), [1, 1, 1], "Should be Equal [1, 1, 1]")
        n.addNeuralTo(1)
        self.assertEqual(n.structure(), [1, 2, 1], "Should be Equal [1, 2, 1]")
        n.addNeuralTo(1)
        self.assertEqual(n.structure(), [1, 3, 1], "Should be Equal [1, 3, 1]")
        n.addInputNeural()
        self.assertEqual(n.structure(), [2, 3, 1], "Should be Equal [2, 3, 1]")
        n.addOutputNeural()
        self.assertEqual(n.structure(), [2, 3, 2], "Should be Equal [2, 3, 2]")
        n.addLayer()
        self.assertEqual(n.structure(), [2, 3, 1, 2], "Should be Equal [2, 3, 1, 2]")

    def testNeuralFeedForward(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            self.assertEqual(n.feedForward(inputs),
                             sigmoid(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))])),
                             "Should be Equal")

    def testNeuralSetFunction(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            n.setFunctionActivation(n.linear)
            self.assertEqual(float('{:.7f}'.format(n.feedForward(inputs))),
                             float('{:.7f}'.format(
                                 linear(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))])))),
                             "Should be Equal")


if __name__ == '__main__':
    unittest.main()
