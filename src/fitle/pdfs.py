"""pdfs.py
=========
Pre-built probability density functions for fitting.

This module provides commonly used PDFs as Model factories. Each
function returns a normalized probability density that can be used
directly with cost functions.

Available PDFs:

- **gaussian**: Normal distribution
- **exponential**: Exponential decay
- **crystalball**: Crystal Ball function (Gaussian core + power-law tail)

Example::

    from fitle import Param, fit, Cost
    from fitle.pdfs import gaussian

    # Default parameters (auto-created)
    model = gaussian()

    # Custom parameters
    mu = Param("mu")(5.0)(0, 10)  # Bounded mean
    sigma = Param.positive("sigma")(1.0)
    model = gaussian(mu, sigma)

    result = fit(model | Cost.NLL(data))
"""
from __future__ import annotations

import numpy as np
from .model import Model, INPUT, const, indecise, Reduction
from .param import Param, index
from .mnp import exp, where, sum, log
import math

SQRT2PI = np.sqrt(2 * np.pi)

def _erf(x):
    """Wrapper for math.erf that Numba can compile."""
    return math.erf(x)
_erf.__name__ = 'erf'


def gaussian(mu: Param | float | None = None, sigma: Param | float | None = None) -> Model:
    """Create a Gaussian (normal) probability density function.

    Returns the normalized PDF: ``(1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)``

    Parameters
    ----------
    mu : Param | float | None, optional
        Mean of the distribution. If None, creates ``Param("mu")``.
    sigma : Param | float | None, optional
        Standard deviation. If None, creates ``Param.positive("sigma")``.

    Returns
    -------
    Model
        A normalized Gaussian PDF model.

    Examples
    --------
    >>> g = gaussian()  # Auto-creates mu and sigma parameters
    >>> g = gaussian(Param("mean")(5.0), Param.positive("width")(1.0))
    """
    mu = mu if mu is not None else Param("mu")
    sigma = sigma if sigma is not None else Param.positive("sigma")

    norm = 1 / (sigma * SQRT2PI)
    arg = -0.5 * ((INPUT - mu) / sigma) ** 2
    return norm * Model(np.exp, [arg])


def exponential(tau: Param | float | None = None) -> Model:
    """Create an exponential probability density function.

    Returns the normalized PDF: ``(1 / tau) * exp(-x / tau)``

    Parameters
    ----------
    tau : Param | float | None, optional
        Decay constant (mean lifetime). If None, creates ``Param.positive("tau")``.

    Returns
    -------
    Model
        A normalized exponential PDF model.

    Examples
    --------
    >>> e = exponential()  # Auto-creates tau parameter
    >>> e = exponential(Param.positive("lifetime")(2.5))
    """
    tau = tau if tau is not None else Param.positive('tau')
    return (1 / tau) * exp(-INPUT / tau)

def crystalball(alpha, n, mu, sigma):
    """
    Crystal Ball PDF.

    alpha > 0 : transition point in units of sigma (typical: 0.5-5)
    n > 1     : tail power (typical: 1.5-10)
    mu        : mean
    sigma     : width
    """
    import warnings

    # Warn about extreme parameter values that may cause numerical issues
    def _check_param(p, name, min_val, max_val):
        if isinstance(p, Param):
            val = p.start if p.start is not None else p.value
            if val is not None and (val < min_val or val > max_val):
                warnings.warn(
                    f"crystalball {name}={val} is outside typical range [{min_val}, {max_val}]. "
                    f"Extreme values may cause numerical issues.",
                    UserWarning, stacklevel=2
                )

    _check_param(alpha, 'alpha', 0.1, 10)
    _check_param(n, 'n', 1.01, 50)

    x = INPUT
    t = (x - mu) / sigma          # (x - μ)/σ

    pref      = n / alpha     # n/|α|
    B         = pref - alpha  # n/|α| - |α|
    C = n / (alpha * (n - 1)) * exp(-0.5 * alpha**2)
    D = np.sqrt(0.5*np.pi) * (1 + Model(_erf, [alpha/np.sqrt(2)]))

    N = 1.0 / (sigma * (C + D))

    # --- core Gaussian part ---
    core = exp(-0.5 * t**2)

    # --- tail: computed fully in log-space ---
    # logA = n*log(n/|α|) - α²/2
    logA      = n * log(pref) - 0.5 * alpha**2
    log_tail  = logA - n * log(B - t)
    tail      = exp(log_tail)

    # --- piecewise: Gaussian for t > -α, tail otherwise ---
    return where(t > -alpha, N * core, N * tail)


def convolve(
    d_x: np.ndarray,
    c: np.ndarray,
    mass_mother: Param | float,
    mu: Param | float,
    sigma: Param | float,
    idx: Param | None = None
) -> Model:
    """Create a convolution of a kernel with coefficient weights.

    Computes a weighted sum of Gaussians at different centers,
    useful for modeling resolution effects or detector response.

    Parameters
    ----------
    d_x : ndarray
        Center positions for each Gaussian component.
    c : ndarray
        Weights (coefficients) for each Gaussian component.
    mass_mother : Param | float
        Reference mass for the shift.
    mu : Param | float
        Mean offset parameter.
    sigma : Param | float
        Common width for all Gaussian components.
    idx : Param, optional
        INDEX parameter for the sum. If None, creates one automatically.

    Returns
    -------
    Model
        A normalized convolution model.
    """
    i = index(len(c)) if not idx else idx
    w = indecise(const(c), i)
    centers = indecise(const(d_x), i)
    shifted_x = INPUT + mass_mother - mu
    g = gaussian(centers, sigma) % shifted_x
    weighted = w * g
    ret = Reduction(weighted, i)
    Xi = indecise(INPUT)
    return ret / ((Xi[1]-Xi[0]) * sum(ret))
