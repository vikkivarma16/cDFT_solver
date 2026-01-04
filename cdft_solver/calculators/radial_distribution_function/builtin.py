# cdft_solver/closures/builtin.py

import numpy as np
from .utils import safe_exp


def py_closure(r, gamma, u, sigma_ab=None):
    return (1.0 + gamma) * (safe_exp(- u) - 1.0)


def hnc_closure(r, gamma, u, sigma_ab=None):
    print("HI I am being accessed")
    return safe_exp(- u + gamma) - gamma - 1.0


def hybrid_closure(r, gamma, u, sigma_ab):
    core = r < sigma_ab
    c_py = py_closure(r, gamma, u)
    c_hnc = hnc_closure(r, gamma, u)
    return np.where(core, c_py, c_hnc)

