import numpy as np
import os


def decorCreateStructure(func):
    def wrapper(cls, structure):
        if len(structure) < 2:
            return False

        for i in structure[1: -1]:
            if i <= 0:
                return False
        return func(cls, structure)

    return wrapper


def decorRemoveLayer(func):
    def wrapper(cls, start: int, end: int = None):
        if end is None:
            end = start + 1

        if cls.layers() <= 2 or end - start - 1 >= cls.layers() - 2:
            return False

        if start < 0 or start >= cls.layers():
            return False

        if end <= start or end > cls.layers():
            return False

        return func(cls, start, end)

    return wrapper


def decorLayer(func):
    def wrapper(cls, layer):
        if layer < 0 or layer >= cls.layers():
            return False
        return func(cls, layer)

    return wrapper


def decorNeural(func):
    def wrapper(cls, layer, index):
        if layer < 0 or layer >= cls.layers():
            return False
        if index < 0 or index >= cls.lenLayer(layer):
            return False
        return func(cls, layer, index)

    return wrapper


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