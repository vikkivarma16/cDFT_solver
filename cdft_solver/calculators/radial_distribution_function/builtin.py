# cdft_solver/closures/builtin.py

import numpy as np
from .utils import safe_exp


    
def py_closure(r, gamma, u, sigma_ab=None):
    """
    Percus-Yevick (PY) closure.
    
    Parameters
    ----------
    r : ndarray
        Distance array
    gamma : ndarray
        Current gamma(r)
    u : ndarray
        Pair potential u(r)
    sigma_ab : float, optional
        Hard-core diameter; if provided, enforce g(r<sigma)=0
    
    Returns
    -------
    c_r : ndarray
        Direct correlation function c(r) via PY closure
    """
    c_r = np.zeros_like(r)

    if sigma_ab is None or sigma_ab ==0:
        # No hard-core, just standard PY closure
        c_r = (1.0 + gamma) * (safe_exp(-u) - 1.0)
    else:
        # Vectorized implementation
        mask_hc = r < sigma_ab   # r < sigma
        mask_out = ~mask_hc      # r >= sigma

        # Inside hard-core: g=0 -> gamma=-1 -> c=-1? But PY sets gamma=-1
        c_r[mask_hc] = -(1.0 + gamma[mask_hc])

        # Outside hard-core: usual PY formula
        c_r[mask_out] = (1.0 + gamma[mask_out]) * (safe_exp(-u[mask_out]) - 1.0)
        
    #c_r = (1.0 + gamma) * (safe_exp(-u) - 1.0)

    return c_r



def hnc_closure(r, gamma, u, sigma_ab=None):
    #print("HI I am being accessed")
    
    
    value = np.exp(-u + gamma) - gamma - 1.0
    
    return value


def hybrid_closure(r, gamma, u, sigma_ab):
    core = r < sigma_ab
    c_py = py_closure(r, gamma, u)
    c_hnc = hnc_closure(r, gamma, u)
    return np.where(core, c_py, c_hnc)

