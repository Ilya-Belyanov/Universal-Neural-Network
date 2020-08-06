import numpy as np


def decorFeedForward(func):
    def wrapper(cls, inputs):
        if type(inputs) == type([]):
            inputs = np.array(inputs)
        res = func(cls, inputs)
        return res

    return wrapper


def decorCreateStructure(func):
    def wrapper(cls, structure):
        if len(structure) < 2:
            return False

        for i in structure[1: -1]:
            if i <= 0:
                return False
        return func(cls, structure)

    return wrapper


def decorTraining(func):
    def wrapper(cls, generation, trainingInput, trainingOutput, count: int = 100):
        if type(trainingInput) == type([]):
            trainingInput = np.array(trainingInput)

        if type(trainingOutput) == type([]):
            trainingOutput = np.array(trainingOutput)

        try:
            m = trainingInput.shape[1]
            n = trainingInput.shape[0]
        except IndexError:
            trainingInput.shape = (1, trainingInput.shape[0])

        try:
            m = trainingOutput.shape[1]
            n = trainingOutput.shape[0]
        except IndexError:
            trainingOutput.shape = (1, trainingOutput.shape[0])

        res = func(cls, generation, trainingInput, trainingOutput, count)
        return res

    return wrapper
