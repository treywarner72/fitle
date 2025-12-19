import numpy as np
import iminuit
from .model import Model, const
from .param import Param
import matplotlib.pyplot as plt


class FitResult:
    def __init__(self, model, m):
        self.minimizer = m
        self.model = model
        self.cost = model.memory['cost'] if "cost" in model.memory else "No memory"
        self.bin_widths = self.cost.bin_widths() if self.cost != "No memory" else 1
        self.predict = model.memory['base'].freeze() * const(self.bin_widths) if "base" in model.memory else "No memory"
        if self.cost != "No memory":
            self.x = self.cost.x()
            self.y = self.cost.y()
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
        plt.errorbar(self.cost.x(), self.cost.y(), linestyle='', marker='.', color='black', yerr=np.sqrt(self.cost.y()))
        
    def plot_fit(self):
        plt.plot(self.cost.x(), self.predict(self.cost.x()))
    
    def dof(self):
        return len(self.cost.x()) - len(self.values) - 1

def fit(model, numba=True, grad=True, ncall = 9999999, options={}):
    if not isinstance(model, Model) or not callable(model):
        raise TypeError("expected a Model instance")

    # Try gradient construction
    grad_model = None
    if grad:
        try:
            grad_model = model.grad()
        except Exception as e:
            print(f"[fit] Warning: gradient disabled ({e})")
            grad = False

    # Try compilation
    if numba:
        try:
            model.compile()
            if grad and grad_model is not None:
                grad_model.compile()
        except Exception as e:
            print(f"[fit] Warning: numba compile disabled ({e})")
            numba = False


    params = model.params

    def loss_fn(*theta):
        for p, v in zip(params, theta):
            p.value = v
        return model.eval(0,0)
        #return sum(model.call(0))

    def grad_fn(*theta):
        for p, v in zip(params, theta):
            p.value = v
        return grad_model.eval(None, None)

    start = [p.start for p in params]
    bounds = [(p.min, p.max) for p in params]

    m = iminuit.Minuit(loss_fn, *start, grad=grad_fn if grad else None, **options)
    m.limits = bounds
    m.migrad(ncall)

    return FitResult(model, m)
