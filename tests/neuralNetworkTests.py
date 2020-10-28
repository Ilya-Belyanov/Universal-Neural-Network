import unittest

from neuralNetwork import *
from src.functions import *


class NeuralNetworkTest(unittest.TestCase):
    def testCalculate(self):
        n = NeuralNetwork()
        n.layers.addInputNeural()
        n.layers.addInputNeural()
        n.layers.addLayers()

        r1 = sigmoid(n.layers.neural(1, 0).weight(0) + n.layers.neural(1, 0).weight(1) + n.layers.neural(1, 0).weight(2))
        result = sigmoid(r1 * n.layers.neural(2, 0).weight(0))

        self.assertEqual(result, n.calculate([1, 1, 1])[0][0])

    def testInput(self):
        n = NeuralNetwork()
        n.layers.addInputNeural()
        n.layers.addLayers()

        self.assertEqual(list(n.calculate(np.array([0, 0]))),
                         list(n.calculate([[0, 0]])),
                         "Should be Equal")

        self.assertEqual(list(n.calculate(np.array([1, 1]))),
                         list(n.calculate([1, 1])),
                         "Should be Equal")

        self.assertEqual(list(n.calculate(np.array([[1, 1]]))),
                         list(n.calculate([[1, 1]])),
                         "Should be Equal")

        self.assertEqual(list(n.calculate(np.array([[1, 1], [0, 0]]))),
                         list(n.calculate([[1, 1], [0, 0]])),
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
        n.layers.addInputNeural()
        n.layers.addLayers(2)
        n.layers.addNeural(1)
        n2 = n.copy()
        self.assertEqual(n.structure(), n2.structure())

    def testAdd(self):
        n = NeuralNetwork()
        self.assertTrue(n.layers.addLayers())
        self.assertTrue(n.layers.addLayers())
        self.assertTrue(n.layers.addNeural(0))
        self.assertTrue(n.layers.addNeural(1))
        self.assertTrue(n.layers.addNeural(2))
        self.assertTrue(n.layers.addNeural(3))
        self.assertFalse(n.layers.addNeural(4))

    def testRemove(self):
        n = NeuralNetwork()
        self.assertTrue(n.layers.addLayers())
        self.assertTrue(n.layers.addLayers())

        self.assertTrue(n.layers.removeLayer(1))
        self.assertFalse(n.layers.removeLayer(4))

        self.assertTrue(n.layers.deleteNeural(1, 0))
        self.assertFalse(n.layers.deleteNeural(6, 0))

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
        n.layers.addLayers()
        self.assertEqual(n.structure(), [1, 1, 1], "Should be Equal [1, 1, 1]")
        n.layers.addNeural(1)
        self.assertEqual(n.structure(), [1, 2, 1], "Should be Equal [1, 2, 1]")
        n.layers.addNeural(1)
        self.assertEqual(n.structure(), [1, 3, 1], "Should be Equal [1, 3, 1]")
        n.layers.addInputNeural()
        self.assertEqual(n.structure(), [2, 3, 1], "Should be Equal [2, 3, 1]")
        n.layers.addOutputNeural()
        self.assertEqual(n.structure(), [2, 3, 2], "Should be Equal [2, 3, 2]")
        n.layers.addLayers()
        self.assertEqual(n.structure(), [2, 3, 1, 2], "Should be Equal [2, 3, 1, 2]")

    def testSave(self):
        n = NeuralNetwork()
        n.layers.addInputNeural()
        n.layers.addInputNeural()
        file = os.getcwd() + '/neural.nn'
        self.assertTrue(n.save(file))

        with open(file) as f:
            self.assertEqual(n.weights(), json.load(f))
        os.remove(file)

        failFile2 = 'dont.txt'
        self.assertFalse(n.save(failFile2))

    def testLoad(self):
        n = NeuralNetwork()
        n.layers.addInputNeural()
        n.layers.addInputNeural()
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
