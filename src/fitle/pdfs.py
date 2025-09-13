import numpy as np
from .model import Model, INPUT, const
from .param import Param

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