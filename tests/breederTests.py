import unittest

from neuralNetwork import NeuralNetwork
from trainers.breeder import Breeder


class BreederTest(unittest.TestCase):
    def setUp(self):
        self.breeder = Breeder()

    def testNetworkCalculate(self):
        NN = NeuralNetwork()
        NN.layers.addInputNeural()
        NN.layers.addInputNeural()

        nn = self.breeder.evolutionLearn(NN, [[0, 0, 1]], [[0]], 10)
        self.assertEqual(round(nn.calculate([[0, 0, 1]])[0][0]), 0)

        nn = self.breeder.evolutionLearn(NN, [[1, 1, 1]], [[1]], 10)
        self.assertEqual(round(nn.calculate([[1, 1, 1]])[0][0]), 1)

        test = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
        testOut = [[0], [1], [1], [0]]

        nn = self.breeder.evolutionLearn(NN, test, testOut, 1000)
        self.assertEqual(round(nn.calculate(test)[0][0]), 0)
        self.assertEqual(round(nn.calculate(test)[1][0]), 1)
        self.assertEqual(round(nn.calculate(test)[2][0]), 1)
        self.assertEqual(round(nn.calculate(test)[3][0]), 0)


if __name__ == '__main__':
    unittest.main()
