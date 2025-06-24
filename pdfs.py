import numpy as np
from .model import Model, X, const
from .param import Param, P, Q, U

SQRT2PI = np.sqrt(2 * np.pi)

def gaussian(x, mu=None, sigma=None):
    """
    Normalized Gaussian PDF:
        f(x; mu, sigma) = 1 / (sigma * sqrt(2pi)) * exp(-0.5 * ((x - mu)/sigma)^2)

    Parameters:
        x      : input variable (symbol or array)
        mu     : mean (symbol or float), default is symbolic P("mu")
        sigma  : std deviation (symbol or float), must be > 0, default is Q("sigma")

    Returns:
        A Model representing the Gaussian PDF
    """
    from numpy import sqrt, pi, exp

    mu = mu if mu is not None else P("mu")
    sigma = sigma if sigma is not None else Q("sigma")

    # Placeholder for now : Implement a symbolic evaluation guard (This should raised if sigma <= 0 during evaluation time)
    # This should be replaced with proper symbolic condition handling
    if hasattr(sigma, "name") and sigma.name == "sigma":
        raise RuntimeError("Symbolic guard for 'sigma > 0' is not yet implemented.")

    norm = 1 / (sigma * sqrt(2 * pi))
    arg = -0.5 * ((x - mu) / sigma) ** 2
    return norm * Model(exp, [arg])

def exponential(x=X, tau=None):
    tau = tau if tau is not None else Q('tau')
    return (1 / tau) * np.e ** (-x / tau)

def uniform(low=None, high=None):
    low = low if isinstance(low, Model) else P[-np.inf, 0, np.inf]('low') if low is None else const(low)
    high = high if isinstance(high, Model) else P[0, 1, np.inf]('high') if high is None else const(high)
    return const(1.0) / (high - low)

def crystal_ball(mu=None, sigma=None, alpha=None, n=None):
    mu = mu if isinstance(mu, Model) else P[-np.inf, 0, np.inf]('mu') if mu is None else const(mu)
    sigma = sigma if isinstance(sigma, Model) else P[0, 1, np.inf]('sigma') if sigma is None else const(sigma)
    alpha = alpha if isinstance(alpha, Model) else P[0, 1, np.inf]('alpha') if alpha is None else const(alpha)
    n = n if isinstance(n, Model) else P[1, 5, np.inf]('n') if n is None else const(n)

    z = (X - mu) / sigma
    abs_alpha = np.abs(alpha)
    sign = np.sign(alpha)

    A = (n / abs_alpha) ** n * np.e ** (-0.5 * abs_alpha ** 2)
    B = n / abs_alpha - abs_alpha

    gauss_core = np.e ** (-0.5 * z ** 2)
    tail = A * (B - sign * z) ** (-n)

    return Model(lambda x, z, a, n: np.where(z * a > -a, np.exp(-0.5 * z**2), (n / a)**n * np.exp(-0.5 * a**2) * (n / a - a - z)**(-n)),
                 [((X - mu) / sigma), alpha, n]) / sigma
