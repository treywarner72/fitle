import numpy as np
from .model import Model

def _wrap(f, name):
    def wrapper(*args):
        def inner(*inner_args):
            return f(*inner_args)
        inner.__name__ = name
        inner._numpy_func = f  # Store original for reference
        return Model(inner, list(args))
    return wrapper

for name in dir(np):
    if name.startswith('_'):
        continue
    f = getattr(np, name)
    if callable(f):
        globals()[name] = _wrap(f, name)
