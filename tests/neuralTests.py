import unittest

from neuralNetwork import *
from functions import sigmoid, linear


class NeuralTest(unittest.TestCase):
    def testNeuralFeedForward(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            self.assertEqual(float('{:.5f}'.format(n.feedForward(inputs))),
                             float('{:.5f}'.format(
                                 sigmoid(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))])))),
                             "Should be Equal")

    def testNeuralSetFunction(self):
        sizes = [1, 2, 3, 10, 100]
        for size in sizes:
            n = Neural(size)
            inputs = [random.randint(0, 10) / 10 for i in range(size)]
            n.setFunctionActivation(linear)
            self.assertEqual(float('{:.7f}'.format(n.feedForward(inputs))),
                             float('{:.7f}'.format(
                                 linear(sum([inputs[i] * n.weights[i][0] for i in range(len(inputs))])))),
                             "Should be Equal")


if __name__ == '__main__':
    unittest.main()
