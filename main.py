import time
import numpy as np

from neuralNetwork import *
from trainers.breeder import Breeder
from trainers.citron import Citron


test = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
testOut = np.array([[0], [1], [1], [0]])

NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()

print("Structure is - ", NN.structure())
breeder = Breeder()

t1 = time.time()
nn = breeder.evolutionLearn(NN, test, testOut, 1000)
t2 = time.time()

print("Result = ", nn.calculate(test))
print("Time work = ", t2 - t1)
print("Loss = ", nn.loss)









