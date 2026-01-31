"""cost.py
=========
Cost function factory for model fitting.

Provides the ``Cost`` class which generates various cost functions
(loss functions) that can be combined with Models using the pipe
operator (``|``).

Supported cost functions:

- **MSE**: Mean Squared Error for regression
- **NLL / unbinnedNLL**: Unbinned Negative Log-Likelihood for PDFs
- **chi2**: Chi-squared for binned histogram fits
- **binnedNLL**: Binned Negative Log-Likelihood for histograms

Example usage::

    from fitle import Param, Cost
    from fitle.pdfs import gaussian

    # Create a model
    model = gaussian(Param("mu"), Param.positive("sigma"))

    # Pipe to a cost function
    cost_model = model | Cost.NLL(data)

    # Fit
    result = fit(cost_model)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .model import Model, const
from .param import INPUT

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

        # Validate array lengths
        if len(x_data) == 0:
            raise ValueError("x cannot be empty.")
        if y_data is not None:
            if len(x_data) != len(y_data):
                raise ValueError(
                    f"x and y must have the same length.\n"
                    f"  Got x: {len(x_data)} elements, y: {len(y_data)} elements."
                )
        if bin_widths is not None:
            if len(bin_widths) != len(x_data):
                raise ValueError(
                    f"bin_widths must have the same length as x.\n"
                    f"  Got x: {len(x_data)} elements, bin_widths: {len(bin_widths)} elements."
                )

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

    def __ror__(self, model: Model) -> Model:
        """Combine a Model with this cost function using the pipe operator.

        When using ``model | cost``, this method is called to create
        a new Model representing the cost function applied to the model.

        Parameters
        ----------
        model : Model
            The model (typically a PDF) to compute cost for.

        Returns
        -------
        Model
            A new Model that computes the cost when evaluated.

        Note
        ----
        The cost model preserves INPUT in its structure rather than
        substituting data. This enables compilation reuse across different
        datasets. The data is stored in model.memory['_eval_x'] and passed
        at evaluation time.
        """
        # Ensure model broadcasts with data by adding INPUT dependency if missing
        if INPUT not in model.free:
            model = model + 0 * INPUT
        ret = self.fcn(model)
        ret.memory['base'] = model
        ret.memory['cost'] = self
        # Store raw data for evaluation (not const-wrapped)
        ret.memory['_eval_x'] = self.x() if hasattr(self.x, '__call__') else self.x
        return ret

    @classmethod
    def MSE(cls, x: NDArray | list, y: NDArray | list) -> Cost:
        """Create a Mean Squared Error cost function.

        Computes ``np.sum((y - model(x))^2)``.

        Parameters
        ----------
        x : array_like
            Independent variable data (input values).
        y : array_like
            Dependent variable data (observed values).

        Returns
        -------
        Cost
            A Cost instance configured for MSE.

        Note
        ----
        The model's INPUT is preserved (not substituted with data).
        Data is passed at evaluation time for compilation reuse.
        """
        cost_instance = cls(x, y)
        # Keep INPUT in model structure - y is embedded as constant
        cost_instance.fcn = lambda model: np.sum((cost_instance.y - model)**2)
        return cost_instance

    @classmethod
    def unbinnedNLL(cls, x: NDArray | list) -> Cost:
        """Create an unbinned Negative Log-Likelihood cost function.

        Computes ``np.sum(-np.log(model(x)))``, suitable for fitting
        probability density functions to unbinned data.

        Parameters
        ----------
        x : array_like
            Observed data points (unbinned).

        Returns
        -------
        Cost
            A Cost instance configured for unbinned NLL.

        Note
        ----
        The model's INPUT is preserved (not substituted with data).
        Data is passed at evaluation time for compilation reuse.
        """
        cost_instance = cls(x)
        # Keep INPUT in model structure - data passed at eval time
        # np.log and np.sum dispatch via __array_ufunc__/__array_function__
        cost_instance.fcn = lambda model: np.sum(-np.log(model))
        return cost_instance

    NLL = unbinnedNLL  # Alias for convenience

    @classmethod
    def chi2(
        cls,
        data: NDArray | list | None = None,
        bins: int | NDArray | None = None,
        range: tuple[float, float] | None = None,
        x: NDArray | list | None = None,
        y: NDArray | list | None = None,
        bin_widths: NDArray | list | None = None,
        zero_method: str = 'error'
    ) -> Cost:
        """Create a chi-squared cost function for binned data.

        Computes ``np.sum((y - model(x) * bin_widths)^2 / y)``.

        Can be initialized either from raw data (which gets binned) or
        from pre-binned histogram data.

        Parameters
        ----------
        data : array_like, optional
            Raw data to bin (use with ``bins``).
        bins : int or array_like, optional
            Number of bins or bin edges (use with ``data``).
        range : tuple[float, float], optional
            Histogram range (use with ``data`` and ``bins``).
        x : array_like, optional
            Bin centers (use with ``y``).
        y : array_like, optional
            Observed counts per bin (use with ``x``).
        bin_widths : array_like, optional
            Width of each bin (required if using ``x`` and ``y``).
        zero_method : str, default 'error'
            How to handle zero counts: 'error' raises, 'absolute' uses
            absolute residuals for zero bins.

        Returns
        -------
        Cost
            A Cost instance configured for chi-squared.
        """
        if data is not None and bins is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot provide both 'data' and 'x,y' for binned cost functions.")
            centers, counts, edges = _bin(data, bins, range)
            widths = np.diff(edges)
            cost_instance = cls(centers, counts, widths)
        elif x is not None and y is not None:
            # Calculate bin widths from centers if not provided
            if bin_widths is None:
                import warnings
                x_arr = np.asarray(x)
                # Calculate widths from spacing between centers
                if len(x_arr) > 1:
                    center_diffs = np.diff(x_arr)
                    # Check if bins are uniformly spaced
                    if not np.allclose(center_diffs, center_diffs[0], rtol=1e-5):
                        warnings.warn(
                            "Inferring bin widths from non-uniform bin centers. "
                            "This may be inaccurate. Consider providing bin_widths explicitly.",
                            UserWarning,
                            stacklevel=2
                        )
                    # Assume first/last bins have same width as neighbors
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

        y_data = cost_instance.y()
        if np.any(y_data == 0):
            if zero_method == 'absolute':
                y_star = const(np.where(y_data > 0, y_data, 1))
                # Keep INPUT - model evaluated at x passed at runtime
                cost_instance.fcn = lambda model: np.sum(
                    ((cost_instance.y - model * cost_instance.bin_widths) ** 2) / y_star
                )
                return cost_instance
            raise ValueError("Chi2 calculation requires that all observed counts (y) be non-zero unless zero_method is set to 'absolute'.")
        # Keep INPUT - model evaluated at x passed at runtime
        cost_instance.fcn = lambda model: np.sum(
            ((cost_instance.y - model * cost_instance.bin_widths) ** 2) / cost_instance.y
        )
        return cost_instance

    @classmethod
    def binnedNLL(
        cls,
        data: NDArray | list | None = None,
        bins: int | NDArray | None = None,
        range: tuple[float, float] | None = None,
        x: NDArray | list | None = None,
        y: NDArray | list | None = None,
        bin_widths: NDArray | list | None = None
    ) -> Cost:
        """Create a binned Negative Log-Likelihood cost function.

        Computes ``2 * np.sum(model(x) * bin_widths - y * np.log(model(x) * bin_widths))``,
        which is the Poisson likelihood for binned data (up to a constant).

        Can be initialized either from raw data (which gets binned) or
        from pre-binned histogram data.

        Parameters
        ----------
        data : array_like, optional
            Raw data to bin (use with ``bins``).
        bins : int or array_like, optional
            Number of bins or bin edges (use with ``data``).
        range : tuple[float, float], optional
            Histogram range (use with ``data`` and ``bins``).
        x : array_like, optional
            Bin centers (use with ``y``).
        y : array_like, optional
            Observed counts per bin (use with ``x``).
        bin_widths : array_like, optional
            Width of each bin (required if using ``x`` and ``y``).

        Returns
        -------
        Cost
            A Cost instance configured for binned NLL.
        """
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

        # Keep INPUT - model evaluated at x passed at runtime
        def _binned_nll_cost(model):
            scaled = model * cost_instance.bin_widths
            return 2 * np.sum(scaled - cost_instance.y * np.log(scaled))

        cost_instance.fcn = _binned_nll_cost
        return cost_instance

def _bin(
    data: NDArray | list,
    bins: int | NDArray,
    range: tuple[float, float] | None = None
) -> tuple[NDArray, NDArray, NDArray]:
    """Bin data into a histogram.

    Parameters
    ----------
    data : array_like
        The input data to be binned.
    bins : int or array_like
        If int, the number of equal-width bins. If array, the bin edges.
    range : tuple[float, float], optional
        The (min, max) range for binning. Defaults to data range.

    Returns
    -------
    centers : ndarray
        The center of each bin.
    counts : ndarray
        The number of data points in each bin (as float64).
    edges : ndarray
        The bin edges.
    """
    counts, edges = np.histogram(data, bins=bins, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts = counts.astype(np.float64)
    return centers, counts, edges
