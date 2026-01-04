# cdft_solver/closures/dispatcher.py

import numpy as np
from .registry import CLOSURE_REGISTRY


def closure_update_c_matrix(
    gamma_r_matrix,
    r,
    pair_closures,
    u_matrix,
    sigma_matrix=None,
    closure_registry=CLOSURE_REGISTRY,
):
    """
    pair_closures[a,b] can be:
      - string: "HNC", "PY", "HYBRID"
      - callable: user-defined closure
    """
    N = gamma_r_matrix.shape[0]
    c_new = np.zeros_like(gamma_r_matrix)

    for i in range(N):
        for j in range(N):

            gamma = gamma_r_matrix[i, j, :]
            u = u_matrix[i, j, :]
            closure = pair_closures[i, j]

            if callable(closure):
                c_new[i, j, :] = closure(
                    r=r,
                    gamma=gamma,
                    u=u,
                    sigma_ab=sigma_matrix[i, j] if sigma_matrix is not None else None,
                )
            else:
                key = closure.upper()
                if key not in closure_registry:
                    raise ValueError(f"Unknown closure '{key}'")

                fn = closure_registry[key]
                c_new[i, j, :] = fn(
                    r=r,
                    gamma=gamma,
                    u=u,
                    sigma_ab=sigma_matrix[i, j] if sigma_matrix is not None else None,
                )

    return c_new

