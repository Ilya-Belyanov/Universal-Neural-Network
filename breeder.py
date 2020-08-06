import numpy as np
import math

from neuralNetworl import NeuralNetwork
from decor import decorTraining


class Breeder:
    countParents = 25
    countChildren = 3

    def evolutionLearn(self, neuralNetwork: NeuralNetwork, trainingInput, trainingOutput, count: int = 100):
        parents = self.createParent(structure=neuralNetwork.structure())
        firstGeneration = self.mutationGeneration(parents)
        return self.training(firstGeneration, trainingInput, trainingOutput, count)

    @staticmethod
    def createParent(structure: list):
        parents = []
        for i in range(Breeder.countParents):
            NN = NeuralNetwork()
            NN.createStructure(structure)
            parents.append(NN)

        return parents

    @staticmethod
    def mutationGeneration(parents: list):
        generation = []
        for p in parents:
            for i in range(Breeder.countChildren):
                NN = p.copy()
                NN.mutation()
                generation.append(NN)
        return generation

    @decorTraining
    def training(self, generation, trainingInputs, trainingOutputs, count: int):

        for i in range(count):
            for network in range(len(generation)):
                result = generation[network].calculate(trainingInputs)
                generation[network].loss += self.checkLoss(result, trainingOutputs)

            generation.sort(key=self.loss)
            generation = generation[:Breeder.countParents]

            if i < count - 1:
                generation = self.mutationGeneration(generation)

        return generation[0]

    @staticmethod
    def checkLoss(result, trainingOutputs):
        error = (result - trainingOutputs) ** 2
        return np.sum(error)

    @staticmethod
    def loss(network):
        return network.loss

