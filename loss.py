import numpy as np
from .param import X
from .model import Model, const
from .mnp import log, sum

class MSE:
    def __init__(self, x,y):
        self.x = const(x)
        self.y = const(y)

    def __ror__(self, model):
        residual = self.y - model % {X: self.x}
        return (residual ** 2).sum()

class NLL:
    def __init__(self, x):
        self.x = const(x)

    def __ror__(self, model):
        return sum(-log(model % {X: self.x}))

def bin(data, bins):
    counts, edges = np.histogram(data, bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts.astype(np.float64)
    return centers, counts

class Chi2:
    def __init__(self, binned):
        self.x = const(binned[0])
        self.y = const(binned[1])

    def __ror__(self, model):
        return Model(np.sum, [((self.y - model % {X: self.x}) ** 2) / self.y])

class BinnedNLL:
    def __init__(self, binned):
        self.x = const(binned[0])
        self.y = const(binned[1])

    def __ror__(self, model):
        return 2 * sum(model % {X: self.x} - self.y * log(model % {X: self.x}))