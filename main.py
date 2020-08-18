import time

from neuralNetworl import *
from breeder import Breeder

test = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
testOut = [[0], [1], [1], [0]]

NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()
NN.addLayer()
print(NN.structure())
breeder = Breeder()
nn = breeder.evolutionLearn(NN, [0, 0, 0], [0], 1000)
print(nn.calculate([0, 0, 0]))
print(nn.loss)











