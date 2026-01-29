import numpy as np
import iminuit
import warnings
from .model import Model, const
from .param import Param
import matplotlib.pyplot as plt


class FitResult:
    def __init__(self, model, m):
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
        else:
            self.bin_widths = 1

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


    def _populate_params(self):
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
    
    def plot_data(self):
        if self.y is None:
            raise ValueError("plot_data() not available for unbinned fits")
        plt.errorbar(self.x, self.y, linestyle='', marker='.', color='black', yerr=np.sqrt(self.y))
        
    def plot_fit(self):
        if self.y is None:
            raise ValueError("plot_data() not available for unbinned fits")
        plt.plot(self.x, self.predict(self.x))
    
    def dof(self):
        if self.y is None:
            raise ValueError("dof() not available for unbinned fits")
        return len(self.cost.x()) - len(self.values) 

def fit(model, numba=True, grad=True, ncall=9999999, options={}):
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

    def loss_fn(*theta):
        for p, v in zip(params, theta):
            p.value = v
        try:
            return model.eval(None, {})
        except Exception as e:
            param_info = ", ".join(f"{p.name}={v:.4g}" for p, v in zip(params, theta))
            raise RuntimeError(
                f"Model evaluation failed during fitting.\n"
                f"  Parameters: {param_info}\n"
                f"  Model: {model}\n"
                f"  Original error: {e}"
            ) from e

    def grad_fn(*theta):
        for p, v in zip(params, theta):
            p.value = v
        try:
            return grad_model.eval(None, {})
        except Exception as e:
            param_info = ", ".join(f"{p.name}={v:.4g}" for p, v in zip(params, theta))
            raise RuntimeError(
                f"Gradient evaluation failed during fitting.\n"
                f"  Parameters: {param_info}\n"
                f"  Original error: {e}"
            ) from e

    start = [p.start for p in params]
    bounds = [(p.min, p.max) for p in params]

    m = iminuit.Minuit(loss_fn, *start, grad=grad_fn if grad else None, **options)
    m.limits = bounds
    m.migrad(ncall)

    return FitResult(model, m)
