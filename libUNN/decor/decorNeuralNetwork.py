import numpy as np
import os


def decorInput(func):
    def wrapper(cls, inputs):
        if isinstance(inputs, list):
            inputs = np.array(inputs)

        try:
            m = inputs.shape[1]
            n = inputs.shape[0]
        except IndexError:
            inputs.shape = (1, inputs.shape[0])

        if inputs.shape[1] == cls.inputs():
            return func(cls, inputs)
        return None

    return wrapper


def decorCreateStructure(func):
    # TODO - polymorph decorator
    def wrapper(cls, structure):
        if len(structure) < 2:
            return False

        for i in structure[1: -1]:
            if i <= 0:
                return False
        return func(cls, structure)

    return wrapper


def decorSave(func):
    def wrapper(cls, file):
        if not isinstance(file, str):
            return False
        fileName, extension = os.path.splitext(file)
        if extension != '.nn':
            return False
        return func(cls, file)

    return wrapper


def decorLoad(func):
    def wrapper(cls, file):
        if os.path.exists(file):
            return func(cls, file)
    return wrapper