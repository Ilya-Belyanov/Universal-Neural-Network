import unittest
import os

from neuralNetwork import *
from src.functions import *


class NeuralNetworkTest(unittest.TestCase):
    def testCalculate(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addInputNeural()
        n.addLayer()

        r1 = sigmoid(n.neural(1, 0).weight(0) + n.neural(1, 0).weight(1) + n.neural(1, 0).weight(2))
        result = sigmoid(r1 * n.neural(2, 0).weight(0))

        self.assertEqual(result, n.calculate([1, 1, 1])[0][0])

    def testInput(self):
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

    def testCopy(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addLayer(2)
        n.addNeuralTo(1)
        n2 = n.copy()
        self.assertEqual(n.structure(), n2.structure())

    def testAdd(self):
        n = NeuralNetwork()
        self.assertTrue(n.addLayer())
        self.assertTrue(n.addLayer())
        self.assertTrue(n.addNeuralTo(0))
        self.assertTrue(n.addNeuralTo(1))
        self.assertTrue(n.addNeuralTo(2))
        self.assertTrue(n.addNeuralTo(3))
        self.assertFalse(n.addNeuralTo(4))

    def testRemove(self):
        n = NeuralNetwork()
        self.assertTrue(n.addLayer())
        self.assertTrue(n.addLayer())
        self.assertTrue(n.removeLayer(1, 3))
        self.assertFalse(n.removeLayer(1, 3))

        self.assertTrue(n.addLayer())
        self.assertTrue(n.addLayer())
        self.assertTrue(n.deleteNeuralFrom(1, 0))
        self.assertFalse(n.removeLayer(1, 0))

    def testSetStructure(self):
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

    def testStructure(self):
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

    def testSave(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addInputNeural()
        file = os.getcwd() + '/neural.nn'
        self.assertTrue(n.save(file))

        with open(file) as f:
            self.assertEqual(n.weights(), json.load(f))
        os.remove(file)

        failFile2 = 'dont.txt'
        self.assertFalse(n.save(failFile2))

    def testLoad(self):
        n = NeuralNetwork()
        n.addInputNeural()
        n.addInputNeural()
        file = os.getcwd() + '/neural.nn'
        self.assertTrue(n.save(file))

        n2 = NeuralNetwork(file)
        self.assertEqual(n2.weights(), n.weights())
        os.remove(file)

        failFile = 'dont.nn'
        failFile2 = 'dont.txt'
        self.assertFalse(n2.load(failFile))
        self.assertFalse(n2.load(failFile2))


if __name__ == '__main__':
    unittest.main()
