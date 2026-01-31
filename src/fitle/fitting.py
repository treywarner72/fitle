"""fitting.py
============
Model fitting using iminuit optimization.

Provides the ``fit()`` function for optimizing Model parameters and
the ``FitResult`` class for storing and visualizing fit results.

The fitting workflow:

1. Build a model with ``Param`` objects
2. Combine with a cost function: ``model | Cost.NLL(data)``
3. Call ``fit(cost_model)`` to optimize
4. Access results via the returned ``FitResult``

Example::

    from fitle import Param, fit, Cost
    from fitle.pdfs import gaussian

    model = gaussian(Param("mu"), Param.positive("sigma"))
    cost = model | Cost.NLL(data)
    result = fit(cost)
    print(result)  # Shows fitted parameters with errors
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import iminuit
import warnings
from .model import Model, const
from .param import Param, _Param
import matplotlib.pyplot as plt


class _FittingContext:
    """Optimized context for repeated model evaluation during fitting.

    Pre-computes and caches everything needed for fast evaluation,
    eliminating per-iteration overhead from hash lookups, property
    walks, and array allocations.
    """

    def __init__(self, model: Model, use_numba: bool = True):
        """Initialize the fitting context.

        Parameters
        ----------
        model : Model
            The model to prepare for fitting. Must be compiled if use_numba=True.
        use_numba : bool
            Whether to use the compiled Numba function.
        """
        self.params = model.params  # Cache once (now also cached in Model)
        self.n_params = len(self.params)
        self.theta = np.empty(self.n_params, dtype=np.float64)  # Reusable buffer
        self.use_numba = use_numba
        self._model = model

        if use_numba:
            # Pre-compute hash and get compiled function
            model_hash = hash(model)
            if model_hash not in Model._compiled_cache:
                raise ValueError("Model must be compiled before creating _FittingContext")
            self._compiled_fn = Model._compiled_cache[model_hash]

            # Pre-compute index params
            self._index_params = [p for p in model.free if p.kind == _Param.index]
            self._has_indices = bool(self._index_params)
        else:
            self._compiled_fn = None
            self._index_params = []
            self._has_indices = False

    def set_and_eval(self, *theta_values, index_map=None):
        """Set params and evaluate in one call.

        Parameters
        ----------
        *theta_values : float
            Parameter values to set.
        index_map : dict, optional
            Mapping of INDEX params to values.

        Returns
        -------
        result
            The model evaluation result.
        """
        # Set parameter values and fill theta buffer
        for i, v in enumerate(theta_values):
            self.params[i].value = v
            self.theta[i] = v

        if self.use_numba:
            index_list = None
            if self._has_indices and index_map:
                index_list = [index_map[p] for p in self._index_params]
            return self._compiled_fn(None, index_list, self.theta)
        else:
            return self._model.eval_py(None, index_map or {})


class FitResult:
    """Container for fit results from ``fit()``.

    Stores the fitted model, parameter values and errors, and provides
    methods for visualization and analysis.

    Attributes
    ----------
    model : Model
        The cost model that was fitted.
    minimizer : iminuit.Minuit
        The underlying iminuit Minuit object.
    fval : float
        The minimum cost function value achieved.
    success : bool
        True if the fit converged successfully.
    values : dict[str, float]
        Fitted parameter values by name.
    errors : dict[str, float]
        Parameter uncertainties by name.
    x : ndarray
        The x-data (bin centers for binned fits).
    y : ndarray | None
        The y-data (counts for binned fits, None for unbinned).
    predict : Model
        The frozen model for predictions, scaled by bin widths.

    Examples
    --------
    >>> result = fit(model | Cost.NLL(data))
    >>> print(result)  # Shows all parameters
    >>> result.plot_data()
    >>> result.plot_fit()
    """

    def __init__(self, model: Model, m: iminuit.Minuit):
        """Initialize a FitResult from a model and Minuit object.

        Parameters
        ----------
        model : Model
            The cost model that was minimized.
        m : iminuit.Minuit
            The Minuit minimizer after optimization.
        """
        self.minimizer = m
        self.model = model

        if "cost" not in model.memory:
            raise ValueError(
                f"Model was not created with a Cost function.\n"
                f"  Use: model | Cost.NLL(data) or model | Cost.MSE(x, y)\n"
                f"  Got: {model}"
            )
        self.cost = model.memory['cost']

        if self.cost.bin_widths is not None:
            bw = self.cost.bin_widths()
            self.bin_widths = bw if bw is not None else 1
            self.bin_width = np.mean(self.bin_widths)
        else:
            self.bin_widths = 1
            self.bin_width = 1

        if "base" not in model.memory:
            raise ValueError(
                f"Model has no base expression stored.\n"
                f"  This usually means the Cost was not applied correctly."
            )
        self.predict = model.memory['base'].freeze() * const(self.bin_widths)
        self.x = self.cost.x()
        self.y = self.cost.y() if self.cost.y is not None else None
        self.fval = m.fval
        self.success = m.valid
        self._populate_params()

        self.values = {}
        self.errors = {}

        i=0
        seen = {}
        for p in model.params:
            base = p.name if p.name else f"x{i}"
            if base in seen:
                seen[base] += 1
                name = f"{base}_{seen[base]}"
            else:
                seen[base] = 0
                name = base

            i += 1
            self.values[name] = p.value
            self.errors[name] = p.error


    def _populate_params(self) -> None:
        """Copy fitted values and errors from Minuit to Param objects.

        Updates each Param's ``value`` and ``error`` attributes with
        the results from the minimizer.
        """
        for p, v, e in zip(self.model.params, self.minimizer.values, self.minimizer.errors):
            p.value = v
            p.error = e

    def __repr__(self):
        i=0
        ret = f"<FitResult fval={self.fval:.3f}, success={self.success}>\n"
        for p in self.model.params:
            name = p.name if p.name else f"x{i}"; i+=1
            ret += f"{name}: {p.value:.4g} ± {p.error:.2g}\n"
        return ret
    
    def plot_data(self) -> None:
        """Plot the observed data with error bars.

        Displays data points as markers with Poisson error bars
        (sqrt(y)). Only available for binned fits.

        Raises
        ------
        ValueError
            If this is an unbinned fit (no y-data).
        """
        if self.y is None:
            raise ValueError("plot_data() not available for unbinned fits")
        plt.errorbar(self.x, self.y, linestyle='', marker='.', color='black', yerr=np.sqrt(self.y))

    def plot_fit(self) -> None:
        """Plot the fitted model prediction.

        Displays the model evaluated at the bin centers, scaled by
        bin widths. Only available for binned fits.

        Raises
        ------
        ValueError
            If this is an unbinned fit (no y-data).
        """
        if self.y is None:
            raise ValueError("plot_fit() not available for unbinned fits")
        plt.plot(self.x, self.predict(self.x))

    def dof(self) -> int:
        """Calculate degrees of freedom.

        Returns the number of data points minus the number of
        fitted parameters. Only available for binned fits.

        Returns
        -------
        int
            Degrees of freedom (n_bins - n_parameters).

        Raises
        ------
        ValueError
            If this is an unbinned fit.
        """
        if self.y is None:
            raise ValueError("dof() not available for unbinned fits")
        # Count only free parameters (not fixed constants where min == max)
        n_free = sum(1 for p in self.model.params if p.min != p.max)
        return len(self.cost.x()) - n_free 

def fit(
    model: Model,
    numba: bool = True,
    grad: bool = True,
    ncall: int = 9999999,
    options: dict | None = None
) -> FitResult:
    """Fit a cost model by minimizing its parameters.

    Uses iminuit's MIGRAD algorithm to find parameter values that
    minimize the cost function.

    Parameters
    ----------
    model : Model
        A cost model created by piping a model to a Cost function,
        e.g., ``gaussian() | Cost.NLL(data)``.
    numba : bool, default True
        If True, compile the model with Numba for faster evaluation.
        Falls back to Python if compilation fails.
    grad : bool, default True
        If True, compute analytical gradients for the optimizer.
        Falls back to numerical gradients if symbolic differentiation
        fails.
    ncall : int, default 9999999
        Maximum number of function calls for MIGRAD.
    options : dict, optional
        Additional keyword arguments passed to ``iminuit.Minuit()``.

    Returns
    -------
    FitResult
        Object containing fitted parameters, errors, and plotting methods.

    Raises
    ------
    TypeError
        If ``model`` is not a callable Model instance.

    Examples
    --------
    >>> model = gaussian(Param("mu"), Param.positive("sigma"))
    >>> cost = model | Cost.NLL(data)
    >>> result = fit(cost)
    >>> print(result.values)  # {'mu': 5.2, 'sigma': 1.1}
    """
    if options is None:
        options = {}

    if not isinstance(model, Model) or not callable(model):
        raise TypeError(
            f"Expected a Model instance, got {type(model).__name__}.\n"
            f"  Create a model using Param, Cost, and arithmetic operations."
        )

    # Try gradient construction
    grad_model = None
    if grad:
        try:
            grad_model = model.grad()
        except Exception as e:
            warnings.warn(
                f"Gradient computation failed, falling back to numerical gradients.\n"
                f"  Reason: {e}\n"
                f"  Model: {model}\n"
                f"  Tip: Use grad=False to suppress this warning.",
                UserWarning,
                stacklevel=2
            )
            grad = False

    # Try compilation
    if numba:
        try:
            model.compile()
            if grad and grad_model is not None:
                grad_model.compile()
        except Exception as e:
            warnings.warn(
                f"Numba compilation failed, falling back to Python.\n"
                f"  Reason: {e}\n"
                f"  Model: {model}\n"
                f"  Tip: Use numba=False to suppress this warning.",
                UserWarning,
                stacklevel=2
            )
            numba = False

    params = model.params
    if not params:
        raise ValueError("Model has no parameters to fit.")

    # Create optimized fitting contexts for fast evaluation
    ctx = _FittingContext(model, use_numba=numba)
    grad_ctx = _FittingContext(grad_model, use_numba=numba) if grad and grad_model else None

    def loss_fn(*theta):
        try:
            return ctx.set_and_eval(*theta)
        except Exception as e:
            param_info = ", ".join(f"{p.name}={v:.4g}" for p, v in zip(params, theta))
            raise RuntimeError(
                f"Model evaluation failed during fitting.\n"
                f"  Parameters: {param_info}\n"
                f"  Model: {model}\n"
                f"  Original error: {e}"
            ) from e

    def grad_fn(*theta):
        try:
            return grad_ctx.set_and_eval(*theta)
        except Exception as e:
            param_info = ", ".join(f"{p.name}={v:.4g}" for p, v in zip(params, theta))
            raise RuntimeError(
                f"Gradient evaluation failed during fitting.\n"
                f"  Parameters: {param_info}\n"
                f"  Original error: {e}"
            ) from e

    start = [p.start for p in params]
    bounds = [(p.min, p.max) for p in params]

    m = iminuit.Minuit(loss_fn, *start, grad=grad_fn if grad_ctx else None, **options)
    m.limits = bounds
    m.migrad(ncall)

    return FitResult(model, m)
