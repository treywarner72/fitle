"""mnp.py
========
Model-aware NumPy function wrappers.

This module automatically wraps all NumPy functions so they can be
used directly with Model objects. When called with Model arguments,
these wrappers create new Model nodes instead of evaluating immediately.

This allows natural syntax like::

    from fitle.mnp import exp, log, sum, where

    model = exp(-x**2)  # Creates a Model, not a number
    model = where(x > 0, x, -x)  # Piecewise Model

All standard NumPy functions are available (exp, log, sin, cos, sum,
where, sqrt, etc.) and work transparently with both Models and regular
arrays.
"""
from __future__ import annotations

import numpy as np
from .model import Model


def _wrap(f, name: str):
    """Wrap a NumPy function to work with Model arguments.

    Creates a wrapper that, when called with any arguments, returns
    a Model node that will call the original function during evaluation.

    Parameters
    ----------
    f : callable
        The NumPy function to wrap.
    name : str
        The function name (used for repr and code generation).

    Returns
    -------
    callable
        A wrapper function that creates Model nodes.
    """
    def wrapper(*args):
        def inner(*inner_args):
            return f(*inner_args)
        inner.__name__ = name
        inner._numpy_func = f  # Store original for reference
        return Model(inner, list(args))
    return wrapper


# Dynamically wrap all callable NumPy functions
for name in dir(np):
    if name.startswith('_'):
        continue
    f = getattr(np, name)
    if callable(f):
        globals()[name] = _wrap(f, name)
