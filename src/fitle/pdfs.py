import numpy as np
from .model import Model, INPUT, const, indecise, Reduction
from .param import Param, index
from .mnp import exp, where, sum
import scipy

SQRT2PI = np.sqrt(2 * np.pi)

def gaussian(mu=None, sigma=None):
    mu = mu if mu is not None else Param("mu")
    sigma = sigma if sigma is not None else Param.positive("sigma")

    norm = 1 / (sigma * 2.5066282746310002)
    arg = -0.5 * ((INPUT - mu) / sigma) ** 2
    return norm * Model(np.exp, [arg])

def exponential(tau=None):
    tau = tau if tau is not None else Param.positive('tau')
    return (1 / tau) * np.e ** (-INPUT / tau)

def crystalball(alpha, n, xbar, sigma):
    x = INPUT
    n_over_alpha = n/alpha
    pexp = exp(-0.5*alpha ** 2)
    A = (n_over_alpha)**n*pexp
    B =  n_over_alpha - alpha
    C = n_over_alpha/(n-1)*pexp
    D = np.sqrt(0.5*np.pi)*(1 + Model(lambda a: scipy.special.erf(a), [alpha/np.sqrt(2)]))
    N = 1/(sigma*(C + D))

    mask = (x - xbar)/sigma > -alpha

    return where((x - xbar)/sigma > -alpha, 
              N*exp(-0.5*((x-xbar)/sigma)**2),
              N*A*(B - (x-xbar)/sigma)**-n
             )


def convolve(d_x, c, mass_mother, mu, sigma):
    i = index(len(c))
    w = indecise(const(c), i)
    centers = indecise(const(d_x), i)
    shifted_x = INPUT + mass_mother - mu
    g = gaussian(centers, sigma) % shifted_x
    weighted = w * g
    ret = Reduction(weighted, i)
    j = index(2)
    iX = indecise(INPUT, j)
    return ret / ((iX[{j:1}]-iX[{j:0}]) * sum(ret))