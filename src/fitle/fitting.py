import numpy as np
import iminuit
from .model import Model
from .param import Param


class FitResult:
    def __init__(self, model, m):
        self.minimizer = m
        self.model = model
        self.predict = model.memory['base'].freeze() if "base" in model.memory else "No memory"
        self.fval = m.fval
        self.success = m.valid
        self._populate_params()

        self.values = {}
        self.errors = {}

        i=0
        for p in model.params:
            name = p.name if p.name else f"x{i}"; i+=1
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

def fit(model, numba=False, grad=False, ncall = 9999999, options={}):
    if not isinstance(model, Model) or not callable(model):
        raise TypeError("expected a Model instance")

     
    if numba == True:
        model.compile()
        if grad:
            grad_model = model.grad().compile()


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
