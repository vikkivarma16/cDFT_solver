import numpy as np


def safe_exp(x, xmin=-500.0, xmax=500.0):
    """
    Numerically safe exponential.
    Clips exponent argument before applying exp.
    """
    return np.exp(np.clip(x, xmin, xmax))
