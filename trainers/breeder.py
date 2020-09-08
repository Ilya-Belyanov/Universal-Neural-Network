import random
import numpy as np

from neuralNetwork import NeuralNetwork
from decor import decorTraining


class Breeder:
    countParents = 10
    countChildren = 3

    def evolutionLearn(self, neuralNetwork: NeuralNetwork, trainingInput, trainingOutput, count: int = 100):
        parents = self.createParent(structure=neuralNetwork.structure())
        firstGeneration = self.mutationGeneration(parents)
        return self.training(firstGeneration, trainingInput, trainingOutput, count)

    @staticmethod
    def createParent(structure: list):
        parents = []
        for i in range(int(Breeder.countParents/2)):
            NN = NeuralNetwork()
            NN.createStructure(structure)
            parents.append(NN)

        for i in range(Breeder.countParents - int(Breeder.countParents/2)):
            NN = parents[i].copy()
            for l in range(1, NN.layers()):
                for n in NN.layer(l):
                    n.weights = -1 * n.weights
            parents.append(NN)
        return parents

    @staticmethod
    def mutationGeneration(parents: list, prob: int = 0.4, k: int = 0.1):
        generation = []
        for p in range(len(parents)):

            for child in range(Breeder.countChildren):
                NN = parents[p].copy()

                for l in range(1, NN.layers()):
                    for i in range(NN.lenLayer(l)):
                        weights = np.copy(NN.layer(l)[i].weights)
                        mutantW = np.zeros(weights.shape)
                        for w in range(weights.shape[0]):
                            if random.random() < prob:
                                mutantW[w][0] = k * (2 * random.random() - 1)
                        NN.layer(l)[i].weights = np.copy(weights + mutantW)

                generation.append(NN)

        return generation

    @staticmethod
    def multiMutationGeneration(parents: list, prob: int = 0.75, k: int = 2):
        generation = []
        for p in range(len(parents)):

            for child in range(Breeder.countChildren):
                NN = parents[p].copy()

                for l in range(1, NN.layers()):
                    for i in range(NN.lenLayer(l)):
                        weights = np.copy(NN.layer(l)[i].weights)
                        mutantW = np.ones(weights.shape)
                        for w in range(mutantW.shape[0]):
                            if random.random() < prob:
                                mutantW[w][0] = k * (2 * random.random() - 1)
                        NN.layer(l)[i].weights = np.copy(weights * mutantW)

                generation.append(NN)

        return generation

    @staticmethod
    def crossingGeneration(parents: list):
        for p in range(len(parents)):
            NN = parents[p].copy()

            index = random.randint(0, len(parents) - 1)
            while index == p:
                index = random.randint(0, len(parents) - 1)
            NNCross = parents[index]

            for l in range(1, NN.layers()):
                for i in range(NN.lenLayer(l)):
                    weights = np.copy(NN.layer(l)[i].weights)

                    changeW = []
                    while len(changeW) != int(weights.shape[0]/2):
                        id = random.randint(0, weights.shape[0] - 1)
                        if id not in changeW:
                            changeW.append(id)

                    for w in range(int(weights.shape[0]/2)):
                        NN.layer(l)[i].weights[changeW[w]][0] = NNCross.layer(l)[i].weights[w][0]
            parents.append(NN)
        return parents

    @decorTraining
    def training(self, generation, trainingInputs, trainingOutputs, count: int):
        oldLoss = 0
        repeat = 0
        for i in range(count):
            for network in generation:
                result = network.calculate(trainingInputs)
                network.loss = self.checkLoss(result, trainingOutputs)

            generation.sort(key=self.loss)
            generation = [generation[i] for i in range(int(Breeder.countParents/2))]

            if generation[0].loss == 0:
                break

            elif generation[0].loss == oldLoss:
                repeat += 1
                if repeat == 5 and i < count - 1:
                    generation = self.multiMutationGeneration(self.crossingGeneration(generation))
                    oldLoss, repeat = 0, 0
                    continue
            else:
                repeat = 0

            oldLoss = generation[0].loss

            if i < count - 1:
                generation = self.mutationGeneration(self.crossingGeneration(generation))

        return generation[0]

    @staticmethod
    def checkLoss(result, trainingOutputs):
        return ((result - trainingOutputs) ** 2).mean()

    @staticmethod
    def loss(network):
        return network.loss
