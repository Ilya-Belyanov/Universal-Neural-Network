import unittest

from neuralNetwork import NeuralNetwork
from trainers.breeder import Breeder


class BreederTest(unittest.TestCase):
    def setUp(self):
        self.breeder = Breeder()

    def testNetworkCalculate(self):
        NN = NeuralNetwork()
        NN.addInputNeural()
        NN.addInputNeural()
        NN.addLayer()

        nn = self.breeder.evolutionLearn(NN, [[0, 0, 1]], [[0]], 10)
        self.assertEqual(round(nn.calculate([[0, 0, 1]])[0][0]), 0)

        nn = self.breeder.evolutionLearn(NN, [[1, 1, 1]], [[1]], 10)
        self.assertEqual(round(nn.calculate([[1, 1, 1]])[0][0]), 1)


if __name__ == '__main__':
    unittest.main()
