import time

from neuralNetwork import *
from trainers.breeder import Breeder


test = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 0]]
testOut = [[0], [1], [1], [0], [0], [1]]

NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()
NN.addLayer()
print("Structure is - ", NN.structure())
breeder = Breeder()

t1 = time.time()
nn = breeder.evolutionLearn(NN, test, testOut, 100)
t2 = time.time()

print("Result = ", nn.calculate(test))
print("Time work = ", t2 - t1)
print("Loss = ", nn.loss)











