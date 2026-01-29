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

    alpha > 0 : transition point in units of sigma
    n         : tail power
    mu        : mean
    sigma     : width
    """

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
