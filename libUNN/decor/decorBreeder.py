import numpy as np


def decorTraining(func):
    def wrapper(cls, generation, trainingInput, trainingOutput, count: int = 100):

        if not isinstance(count, int) or count < 0:
            return False

        if isinstance(trainingInput, list):
            trainingInput = np.array(trainingInput)

        if isinstance(trainingOutput, list):
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