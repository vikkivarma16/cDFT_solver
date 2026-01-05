# cdft_solver/closures/builtin.py

import numpy as np
from .utils import safe_exp


def py_closure(r, gamma, u, sigma_ab=None):
    #print("its me running")
    return (1.0 + gamma) * (safe_exp(- u) - 1.0)


def hnc_closure(r, gamma, u, sigma_ab=None):
    #print("HI I am being accessed")
    
    with np.errstate(over='ignore', invalid='ignore'):
        # attempt the usual closure
        value = np.exp(- u + gamma) - gamma - 1.0

    # handle NaNs or infs due to overflow
    mask = ~np.isfinite(value)  # True for NaN or inf
    if np.any(mask):
        # linearized approximation: exp(gamma) ≈ 1 + gamma
        value = (1 + gamma[mask]) * (np.exp(-  u[mask]) - 1)
    
    return value


def hybrid_closure(r, gamma, u, sigma_ab):
    core = r < sigma_ab
    c_py = py_closure(r, gamma, u)
    c_hnc = hnc_closure(r, gamma, u)
    return np.where(core, c_py, c_hnc)

