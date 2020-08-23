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
        for i in range(Breeder.countParents):
            NN = NeuralNetwork()
            NN.createStructure(structure)
            parents.append(NN)

        return parents

    @staticmethod
    def mutationGeneration(parents: list, prob: int = 0.4, k: int = 0.1):
        return [p.copy().mutation(prob, k) for i in range(Breeder.countChildren) for p in parents]

    @decorTraining
    def training(self, generation, trainingInputs, trainingOutputs, count: int):
        oldLoss = 0
        repeat = 0
        for i in range(count):
            for network in generation:
                result = network.calculate(trainingInputs)
                network.loss = self.checkLoss(result, trainingOutputs)

            generation.sort(key=self.loss)
            generation = [generation[i] for i in range(Breeder.countParents)]

            if generation[0].loss == 0:
                break
            elif generation[0].loss == oldLoss:
                repeat += 1
                if repeat == 5 and i < count - 1:
                    generation = self.mutationGeneration(generation, 1, 20)
                    oldLoss = 0
                    repeat = 0
                    continue
            else:
                repeat = 0

            oldLoss = generation[0].loss

            if i < count - 1:
                generation = self.mutationGeneration(generation)

        return generation[0]

    @staticmethod
    def checkLoss(result, trainingOutputs):
        return ((result - trainingOutputs) ** 2).mean()

    @staticmethod
    def loss(network):
        return network.loss
