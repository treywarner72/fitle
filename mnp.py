import numpy as np
import types
from .model import Model

def _wrap_ufunc(f, name):
    def wrapper(*args):
        def inner(*inner_args):
            return f(*inner_args)
        inner.__name__ = name
        return Model(inner, list(args))
    return wrapper

def _wrap_plain(f, name):
    def wrapper(*args):
        return Model(f, list(args))
    return wrapper

for name in dir(np):
    if name.startswith('_'):
        continue
    f = getattr(np, name)
    if callable(f):
        wrapped = _wrap_ufunc(f, name) if isinstance(f, np.ufunc) else _wrap_plain(f, name)
        globals()[name] = wrapped
