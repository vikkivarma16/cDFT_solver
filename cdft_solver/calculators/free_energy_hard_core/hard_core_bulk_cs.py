def hard_core_bulk_cs(
    ctx=None,
    hc_data=None,
    export_json=True,
    filename="Solution_hardcore.json"
):
    """
    BMCSL hard-sphere free energy for colloids
    + Free Volume Theory for polymers.

    flag = 1 → colloids
    flag = 0 → polymers
    """

    import numpy as np
    import sympy as sp
    import json
    from pathlib import Path

    # -------------------------
    # Input
    # -------------------------
    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

    n = len(species)

    def extract_diagonal(data, n):
        arr = np.asarray(data)
        if arr.ndim == 1 and arr.size == n:
            return arr.tolist()
        if arr.ndim == 2:
            return arr.diagonal().tolist()
        if arr.ndim == 1 and arr.size == n*n:
            return arr.reshape((n, n)).diagonal().tolist()
        raise ValueError("Invalid input")

    sigma = extract_diagonal(sigma_raw, n)
    flag  = extract_diagonal(flag_raw, n)

    sigma = [float(s) for s in sigma]
    flag  = [int(f) for f in flag]

    rho = [sp.symbols(f"rho_{s}") for s in species]

    colloids = [i for i in range(n) if flag[i] == 1]
    polymers = [i for i in range(n) if flag[i] == 0]

    # -------------------------
    # BMCSL moments (colloids only)
    # -------------------------
    xi0 = sum(rho[i] for i in colloids)
    xi1 = sum(rho[i] * sigma[i] for i in colloids) * sp.pi/6
    xi2 = sum(rho[i] * sigma[i]**2 for i in colloids) * sp.pi/6
    xi3 = sum(rho[i] * sigma[i]**3 for i in colloids) * sp.pi/6

    # -------------------------
    # BMCSL free energy density
    # -------------------------
    F_hs = (
        xi0 * sp.log(1 - xi3)
        + (3 * xi1 * xi2) / (1 - xi3)
        + (xi2**3) / (xi3 * (1 - xi3)**2)
    )

    # NOTE: sign convention → usually excess is NEGATIVE log term
    F_hs = -F_hs

    # -------------------------
    # Free Volume Theory
    # -------------------------
    a = [sp.pi * sigma[i]**2 for i in range(n)]
    s = [sigma[i]/2 for i in range(n)]
    v = [sp.pi * sigma[i]**3 / 6 for i in range(n)]

    eta = xi3  # same thing

    sum_rho_a = sum(rho[i] * a[i] for i in colloids)
    sum_rho_s = sum(rho[i] * s[i] for i in colloids)
    sum_rho   = sum(rho[i] for i in colloids)

    F_fv = 0

    for p in polymers:

        A_p = s[p]*sum_rho_a + a[p]*sum_rho_s + v[p]*sum_rho

        B_p = (sp.Rational(1,2) * s[p]**2 * sum_rho_a**2
               + v[p] * sum_rho_s * sum_rho_a)

        C_p = (1/(12*sp.pi)) * v[p] * sum_rho_a**3

        alpha_p = (1 - eta) * sp.exp(
            - A_p/(1 - eta)
            - B_p/(1 - eta)**2
            - C_p/(1 - eta)**3
        )

        F_fv += -rho[p] * sp.log(alpha_p)

    # -------------------------
    # Total free energy
    # -------------------------
    F = F_hs + F_fv

    F_func = sp.Lambda(tuple(rho), F)

    result = {
        "variables": tuple(rho),
        "function": F_func,
        "expression": F
    }

    # -------------------------
    # Export
    # -------------------------
    if export_json and ctx is not None:
        out_file = Path(ctx.scratch_dir) / filename
        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "sigma": sigma,
                    "flag": flag,
                    "expression": str(F),
                },
                f,
                indent=2,
            )

    return result
