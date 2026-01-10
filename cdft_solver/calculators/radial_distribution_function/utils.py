import numpy as np


def safe_exp(x, xmin=-50000000.0, xmax=5000000.0):
    """
    Numerically safe exponential.
    Clips exponent argument before applying exp.
    """
    return np.exp(np.clip(x, xmin, xmax))
