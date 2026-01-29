import numpy as np
from .model import Model, INPUT, const, indecise, Reduction
from .param import Param, index
from .mnp import exp, where, sum, log
import scipy

SQRT2PI = np.sqrt(2 * np.pi)

def gaussian(mu=None, sigma=None):
    mu = mu if mu is not None else Param("mu")
    sigma = sigma if sigma is not None else Param.positive("sigma")

    norm = 1 / (sigma * SQRT2PI)
    arg = -0.5 * ((INPUT - mu) / sigma) ** 2
    return norm * Model(np.exp, [arg])

def exponential(tau=None):
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
    D = np.sqrt(0.5*np.pi) * (1 + Model(lambda a: scipy.special.erf(a), [alpha/np.sqrt(2)]))

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


def convolve(d_x, c, mass_mother, mu, sigma, idx=None):
    i = index(len(c)) if not idx else idx
    w = indecise(const(c), i)
    centers = indecise(const(d_x), i)
    shifted_x = INPUT + mass_mother - mu
    g = gaussian(centers, sigma) % shifted_x
    weighted = w * g
    ret = Reduction(weighted, i)
    Xi = indecise(INPUT)
    return ret / ((Xi[1]-Xi[0]) * sum(ret))
