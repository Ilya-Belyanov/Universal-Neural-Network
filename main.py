import time

from neuralNetwork import *
from trainers.breeder import Breeder
from trainers.perceptron import Perceptron2L

test = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
testOut = np.array([[0], [1], [1], [0]])

NN = NeuralNetwork()
NN.layers.addInputNeural()
NN.layers.addInputNeural()

breeder = Breeder()

print("Result before = ", NN.calculate(test), end='\n\n')
t1 = time.time()
nn = breeder.evolutionLearn(NN, test, testOut, 1000)
t2 = time.time()

print("Result after breeder = \n", nn.calculate(test))
print("Time work breeder= ", t2 - t1)
print("Loss = ", nn.loss, end='\n\n')

NN = NeuralNetwork()
NN.layers.addInputNeural()
NN.layers.addInputNeural()


c = Perceptron2L()
t1 = time.time()
nn = c.learn(NN, test, testOut, 1000)
t2 = time.time()
print("Result after perceptron = \n", nn.calculate(test))
print("Time work perceptron= ", t2 - t1)
print(nn.calculate(test))










