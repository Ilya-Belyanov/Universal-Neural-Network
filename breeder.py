import numpy as np

from neuralNetwork import NeuralNetwork
from decor import decorTraining


class Breeder:
    countParents = 15
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
        return [p.copy().mutation() for i in range(Breeder.countChildren) for p in parents]

    @decorTraining
    def training(self, generation, trainingInputs, trainingOutputs, count: int):

        for i in range(count):
            for network in generation:
                result = network.calculate(trainingInputs)
                network.loss = self.checkLoss(result, trainingOutputs)

            generation.sort(key=self.loss)
            generation = [generation[i] for i in range(Breeder.countParents)]

            if i < count - 1:
                generation = self.mutationGeneration(generation)

        return generation[0]

    @staticmethod
    def checkLoss(result, trainingOutputs):
        error = (result - trainingOutputs) ** 2
        return np.sum(error) / error.size

    @staticmethod
    def loss(network):
        return network.loss
