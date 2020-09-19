import time

from neuralNetwork import *
from trainers.breeder import Breeder
from trainers.citron import Citron

test = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
testOut = np.array([[0], [1], [1], [0]])

NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()

breeder = Breeder()

print("Result before = ", NN.calculate(test))
t1 = time.time()
nn = breeder.evolutionLearn(NN, test, testOut, 1000)
t2 = time.time()

print("Result after= ", nn.calculate(test))
print("Time work = ", t2 - t1)
print("Loss = ", nn.loss)

'''
NN = NeuralNetwork()
NN.addInputNeural()
NN.addInputNeural()

c = Citron()
nn = c.learn(NN, test, testOut, 10000)
print(nn.calculate(test))
'''









