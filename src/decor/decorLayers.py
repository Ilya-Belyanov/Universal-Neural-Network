def decorLenLayer(func):
    def wrapper(cls, layer: int):
        if layer < 0 or layer >= len(cls):
            return False
        return func(cls, layer)

    return wrapper


def decorDeleteNeuralFrom(func):
    def wrapper(cls, layer, index):
        if layer < 0 or layer >= len(cls):
            return False
        if index < 0 or index >= cls.lenLayer(layer):
            return False
        return func(cls, layer, index)

    return wrapper


def decorAddNeuralTo(func):
    def wrapper(cls, layer: int, count: int = 1):
        if layer < 0 or layer >= len(cls) or count < 0:
            return False
        return func(cls, layer, count)

    return wrapper
