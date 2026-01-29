import numpy as np
from .model import Model, const
from .param import INPUT
from .mnp import log, sum, where

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
        bin_widths (const, optional): Constant representation of bin widths for binned data.
    """
    def __init__(self, x_data, y_data=None, bin_widths=None):
        # Convert lists to numpy arrays for consistency
        if not isinstance(x_data, np.ndarray):
            x_data = np.asarray(x_data)
        if y_data is not None and not isinstance(y_data, np.ndarray):
            y_data = np.asarray(y_data)
        if bin_widths is not None and not isinstance(bin_widths, np.ndarray):
            bin_widths = np.asarray(bin_widths)

        self.x = const(x_data)
        self.y = const(y_data) if y_data is not None else None
        self.bin_widths = const(bin_widths) if bin_widths is not None else None
        self.fcn = self._base_fcn # Assign the method to the attribute

    def _base_fcn(self, *args):
        """
        Placeholder function for base Cost class.
        Specific cost functions will override the __ror__ behavior.
        """
        raise NotImplementedError("This is a base Cost class. Use specific cost function generators like Cost.MSE, Cost.NLL, etc.")

    def __ror__(self, model):
        # Ensure model broadcasts with data by adding INPUT dependency if missing
        if INPUT not in model.free:
            model = model + 0 * INPUT
        ret = self.fcn(model)
        ret.memory['base'] = model
        ret.memory['cost'] = self
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
    def chi2(cls, data=None, bins=None, range=None, x=None, y=None, bin_widths=None, zero_method='error'):
        if data is not None and bins is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot provide both 'data' and 'x,y' for binned cost functions.")
            centers, counts, edges = _bin(data, bins, range)
            widths = np.diff(edges)
            cost_instance = cls(centers, counts, widths)
        elif x is not None and y is not None:
            # Calculate bin widths from centers if not provided
            if bin_widths is None:
                x_arr = np.asarray(x)
                # Calculate widths from spacing between centers
                if len(x_arr) > 1:
                    # Use differences between adjacent centers
                    center_diffs = np.diff(x_arr)
                    # Assume first and last bins have same width as their neighbors
                    widths = np.empty(len(x_arr))
                    widths[0] = center_diffs[0]
                    widths[-1] = center_diffs[-1]
                    widths[1:-1] = (center_diffs[:-1] + center_diffs[1:]) / 2
                else:
                    raise ValueError("Cannot infer bin widths from a single bin center. Please provide bin_widths explicitly.")
            else:
                widths = bin_widths
            cost_instance = cls(x, y, widths)
        else:
            raise ValueError("For chi2, provide either 'data' and 'bins' (and optional 'range') or 'x', 'y', and 'bin_widths'.")

        if np.any(cost_instance.y() == 0):
            if zero_method == 'absolute':
                condition = cost_instance.y > 0
                y_star = where(cost_instance.y>0, cost_instance.y, 1)
                # Scale model predictions by bin width
                cost_instance.fcn = lambda model: sum(
                    ((cost_instance.y - (model % cost_instance.x) * cost_instance.bin_widths) ** 2) / y_star
                )
                return cost_instance
            raise ValueError("Chi2 calculation requires that all observed counts (y) be non-zero unless zero_method is set to 'absolute'.")
        
        # Scale model predictions by bin width for proper chi-squared
        cost_instance.fcn = lambda model: Model(
            np.sum, 
            [((cost_instance.y - (model % cost_instance.x) * cost_instance.bin_widths) ** 2) / cost_instance.y]
        )
        return cost_instance

    @classmethod
    def binnedNLL(cls, data=None, bins=None, range=None, x=None, y=None, bin_widths=None):
        if data is not None and bins is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot provide both 'data' and 'x,y' for binned cost functions.")
            centers, counts, edges = _bin(data, bins, range)
            widths = np.diff(edges)
            cost_instance = cls(centers, counts, widths)
        elif x is not None and y is not None:
            if bin_widths is None:
                raise ValueError("When providing x and y directly, you must also provide bin_widths.")
            cost_instance = cls(x, y, bin_widths)
        else:
            raise ValueError("For binned_nll, provide either 'data' and 'bins' (and optional 'range') or 'x', 'y', and 'bin_widths'.")

        # Scale model predictions by bin width
        cost_instance.fcn = lambda model: 2 * sum(
            (model % cost_instance.x) * cost_instance.bin_widths - 
            cost_instance.y * log((model % cost_instance.x) * cost_instance.bin_widths)
        )
        return cost_instance

def _bin(data, bins, range=None):
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
               - edges (ndarray): The bin edges.
    """
    counts, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts.astype(np.float64)
    return centers, counts, edges
