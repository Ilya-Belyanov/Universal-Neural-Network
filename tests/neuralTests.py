import unittest
import random

from neuralNetwork import *
from src.functions import sigmoid, linear


class NeuralTest(unittest.TestCase):
    def testFeedForward(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            self.assertEqual(float('{:.5f}'.format(n.feedForward(inputs))),
                             float('{:.5f}'.format(
                                 sigmoid(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))])) + n.bias)),
                             "Should be Equal")

    def testSetFunction(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            n.setFunctionActivation(linear)
            self.assertEqual(float('{:.7f}'.format(n.feedForward(inputs))),
                             float('{:.7f}'.format(
                                 linear(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))]) + n.bias))),
                             "Should be Equal")

    def testCopy(self):
        n = Neural(3)
        n2 = n.copy()
        for i in range(n.weights.shape[0]):
            self.assertEqual(n.weights[i][0], n2.weights[i][0])
        self.assertEqual(n.bias, n2.bias)
        self.assertEqual(n.functionActivation(), n2.functionActivation())


if __name__ == '__main__':
    unittest.main()
