import numpy as np
from .param import X
from .model import Model, const
from .mnp import log, sum

class Cost:
    """
    A unified class for generating and calculating various cost functions for model fitting.

    This class acts as a factory, providing class methods to create specific cost function
    instances (e.g., MSE, NLL, Chi-squared, Binned NLL). Once instantiated, a `Cost` object
    can be piped with a `Model` instance using the `|` operator to create a model that represents the cost.

    Attributes:
        x (const): Constant representation of the independent variable data.
        y (const, optional): Constant representation of the dependent variable data
                             (observed counts/values). `None` for some NLL cases.
    """
    def __init__(self, x_data, y_data=None):
        self.x = const(x_data)
        self.y = const(y_data) if y_data is not None else None
        self.fcn = self._base_fcn # Assign the method to the attribute

    def _base_fcn(self, *args):
        """
        Placeholder function for base Cost class.
        Specific cost functions will override the __ror__ behavior.
        """
        raise NotImplementedError("This is a base Cost class. Use specific cost function generators like Cost.MSE, Cost.NLL, etc.")

    def __ror__(self, model):
        ret = self.fcn(model)
        ret.memory['base'] = model
        return ret

    @classmethod
    def MSE(cls, x, y):
        cost_instance = cls(x, y)
        cost_instance.fcn = lambda model: sum((cost_instance.y - model % cost_instance.x)**2)
        return cost_instance

    @classmethod
    def unbinnedNLL(cls, x):
        cost_instance = cls(x)
        cost_instance.fcn = lambda model: sum(-log(model % cost_instance.x))
        return cost_instance
        
    NLL = unbinnedNLL

    @classmethod
    def chi2(cls, data=None, bins=None, range=None, x=None, y=None):
        if data is not None and bins is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot provide both 'data' and 'x,y' for binned cost functions.")
            centers, counts = bin(data, bins, range)
            cost_instance = cls(centers, counts)
        elif x is not None and y is not None:
            cost_instance = cls(x, y)
        else:
            raise ValueError("For chi2, provide either 'data' and 'bins' (and optional 'range') or 'x' and 'y'.")

        if np.any(cost_instance.y() == 0):
            raise ValueError("Chi2 calculation requires that all observed counts (y) be non-zero.")
        
        cost_instance.fcn = lambda model: Model(np.sum, [((cost_instance.y - model % cost_instance.x) ** 2) / cost_instance.y])
        return cost_instance

    @classmethod
    def binnedNLL(cls, data=None, bins=None, range=None, x=None, y=None):
        if data is not None and bins is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot provide both 'data' and 'x,y' for binned cost functions.")
            centers, counts = bin(data, bins, range)
            cost_instance = cls(centers, counts)
        elif x is not None and y is not None:
            cost_instance = cls(x, y)
        else:
            raise ValueError("For binned_nll, provide either 'data' and 'bins' (and optional 'range') or 'x' and 'y'.")

        cost_instance.fcn = lambda model: 2 * sum(model % cost_instance.x - cost_instance.y * log(model % cost_instance.x))
        return cost_instance

def bin(data, bins, range=None):
    """
    Bins the data.

    Args:
        data (array_like): The input data to be binned.
        bins (int or sequence of scalars): If bins is an int, it defines the number of
                                           equal-width bins in the given range. If bins
                                           is a sequence, it defines the bin edges.
        range (tuple, optional): The lower and upper range of the bins.
                                      If not provided, the range is (data.min(), data.max()).

    Returns:
        tuple: A tuple containing:
               - centers (ndarray): The centers of the bins.
               - counts (ndarray): The number of data points in each bin.
    """
    counts, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts.astype(np.float64)
    return centers, counts
