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
from .model import Model, INPUT, const, _indecise, Reduction
from .param import Param, _Param, index
import math

SQRT2PI = np.sqrt(2 * np.pi)

def _erf(x):
    """Wrapper for math.erf that Numba can compile."""
    return math.erf(x)
_erf.__name__ = 'erf'


def gaussian(mu: _Param | float | None = None, sigma: _Param | float | None = None) -> Model:
    """Create a Gaussian (normal) probability density function.

    Returns the normalized PDF: ``(1 / (sigma * sqrt(2*pi))) * np.exp(-0.5 * ((x - mu) / sigma)^2)``

    Parameters
    ----------
    mu : _Param | float | None, optional
        Mean of the distribution. If None, creates ``Param("mu")``.
    sigma : _Param | float | None, optional
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


def exponential(
    tau: _Param | float | None = None,
    start: float | None = None,
    end: float | None = None
) -> Model:
    """Create an exponential probability density function.

    If start and end are provided, returns a truncated exponential PDF
    normalized over [start, end]:
        ``(1/tau) * exp(-(x-start)/tau) / (1 - exp(-(end-start)/tau))``

    Otherwise returns the standard exponential PDF over [0, inf):
        ``(1/tau) * exp(-x/tau)``

    Parameters
    ----------
    tau : _Param | float | None, optional
        Decay constant (mean lifetime). If None, creates ``Param.positive("tau")``.
    start : float, optional
        Left boundary for truncated exponential.
    end : float, optional
        Right boundary for truncated exponential.

    Returns
    -------
    Model
        A normalized exponential PDF model.

    Examples
    --------
    >>> e = exponential()  # Standard exponential over [0, inf)
    >>> e = exponential(tau=+Param, start=0, end=10)  # Truncated
    """
    tau = tau if tau is not None else Param.positive('tau')
    if start is not None and end is not None:
        norm = 1 - np.exp(-(end - start) / tau)
        return (1 / tau) * np.exp(-(INPUT - start) / tau) / norm
    return (1 / tau) * np.exp(-INPUT / tau)

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
        if isinstance(p, _Param):
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
    C = n / (alpha * (n - 1)) * np.exp(-0.5 * alpha**2)
    D = np.sqrt(0.5*np.pi) * (1 + Model(_erf, [alpha/np.sqrt(2)]))

    N = 1.0 / (sigma * (C + D))

    # --- core Gaussian part ---
    core = np.exp(-0.5 * t**2)

    # --- tail: computed fully in log-space ---
    # logA = n*np.log(n/|α|) - α²/2
    # Clamp B - t to avoid log of negative values (np.where evaluates both branches)
    logA      = n * np.log(pref) - 0.5 * alpha**2
    log_tail  = logA - n * np.log(np.maximum(B - t, 1e-10))
    tail      = np.exp(log_tail)

    # --- piecewise: Gaussian for t > -α, tail otherwise ---
    return np.where(t > -alpha, N * core, N * tail)


def convolve(
    d_x: np.ndarray,
    c: np.ndarray,
    mass_mother: _Param | float,
    mu: _Param | float,
    sigma: _Param | float,
    idx: _Param | None = None
) -> Model:
    """Create a convolution of a kernel with coefficient weights.

    Computes a weighted sum of Gaussians at different centers,
    useful for modeling resolution effects or detector response.
    Returns a normalized PDF that integrates to 1.

    Parameters
    ----------
    d_x : ndarray
        Center positions for each Gaussian component.
    c : ndarray
        Weights (coefficients) for each Gaussian component. Will be
        normalized to sum to 1 internally.
    mass_mother : _Param | float
        Reference mass for the shift.
    mu : _Param | float
        Mean offset parameter.
    sigma : _Param | float
        Common width for all Gaussian components.
    idx : Param, optional
        INDEX parameter for the sum. If None, creates one automatically.

    Returns
    -------
    Model
        A normalized PDF model (integrates to 1).
    """
    i = index(len(c)) if not idx else idx
    # Pre-normalize weights to sum to 1
    c_normalized = np.asarray(c) / np.sum(c)
    w = _indecise(const(c_normalized), i)
    centers = _indecise(const(d_x), i)
    shifted_x = INPUT + mass_mother - mu
    g = gaussian(centers, sigma) % shifted_x
    weighted = w * g
    ret = Reduction(weighted, i)
    return ret
