import time

from neuralNetwork import *
from breeder import Breeder

import time

test = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
testOut = [[0], [1], [1], [0]]

NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()
NN.addLayer()
print(NN.structure(), end="\n")
breeder = Breeder()

t1 = time.time()
nn = breeder.evolutionLearn(NN, [0, 0, 0], [0], 1000)
t2 = time.time()

print(t2 - t1, " Time\n")
print(nn.calculate([0, 0, 0]), end="\n")
print(nn.loss)











