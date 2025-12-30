def free_energy_hard_core(
    ctx=None,
    hc_data=None,
    export_json=True,
    filename="Solution_hardcore.json"
):
    """
    Computes the hard-core contribution to the free energy for a multi-species system.
    """

    import numpy as np
    import sympy as sp
    import json
    from pathlib import Path

    # -------------------------
    # Validate input
    # -------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

    if not species or sigma_raw is None or flag_raw is None:
        raise ValueError("hc_data must contain 'species', 'sigma', and 'flag'")

    n_species = len(species)

    
    
    
    
    def extract_diagonal(data, n, name="array"):
        """
        Extract diagonal elements from:
          - 1D length-n list
          - 2D n×n matrix
          - 1D length-n*n flat array
        """
        arr = np.asarray(data)

        # Case 1: already diagonal vector
        if arr.ndim == 1 and arr.size == n:
            return arr.tolist()

        # Case 2: flat n*n array
        if arr.ndim == 1 and arr.size == n * n:
            return arr.reshape((n, n)).diagonal().tolist()

        # Case 3: square matrix
        if arr.ndim == 2 and arr.shape == (n, n):
            return arr.diagonal().tolist()

        raise ValueError(
            f"{name} must be length-{n}, {n}x{n}, or flat length-{n*n} array"
        )

    sigmai = extract_diagonal(sigma_raw, n_species, name="sigma")
    flag   = extract_diagonal(flag_raw,  n_species, name="flag")

    sigmai = [float(s) for s in sigmai]
    flag   = [int(f) for f in flag]


    # -------------------------
    # Define symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # No hard-core shortcut
    # -------------------------
    if all(s == 0.0 for s in sigmai) and all(f == 0 for f in flag):
        fhc = sp.Integer(0)

    else:
        # -------------------------
        # Hard-core free energy (FMT-like)
        # -------------------------
        def _fhc_expression(sigmai, flag, densities):
            measures = [
                [1, sigma / 2, sp.pi * sigma**2, sp.pi * sigma**3 / 6]
                for sigma in sigmai
            ]

            etas = [sp.symbols(f"eta_{i}") for i in range(len(sigmai))]

            variables = [
                [densities[i] * measures[i][j] for j in range(4)]
                for i in range(len(sigmai))
            ]

            fac1 = 1 - sum(etas)
            fac2 = sum(etas[i] for i in range(len(sigmai)) if flag[i] == 1)

            phi0 = fac1 * sp.log(1 - fac2) + fac2

            diff_1 = [sp.diff(phi0, etas[i]) for i in range(len(sigmai))]
            diff_2 = [
                [sp.diff(phi0, etas[i], etas[j]) for j in range(len(sigmai))]
                for i in range(len(sigmai))
            ]
            diff_3 = [
                [
                    [
                        sp.diff(phi0, etas[i], etas[j], etas[k])
                        for k in range(len(sigmai))
                    ]
                    for j in range(len(sigmai))
                ]
                for i in range(len(sigmai))
            ]

            phi1 = sum(variables[i][0] * diff_1[i] for i in range(len(sigmai)))

            phi2 = sum(
                variables[i][1] * variables[j][2] * diff_2[i][j]
                for i in range(len(sigmai))
                for j in range(len(sigmai))
            )

            phi3 = sum(
                (1 / (24 * sp.pi))
                * variables[i][2]
                * variables[j][2]
                * variables[k][2]
                * diff_3[i][j][k]
                for i in range(len(sigmai))
                for j in range(len(sigmai))
                for k in range(len(sigmai))
            )

            fhc_expr = phi1 + phi2 + phi3

            # Substitute η_i → n3_i
            for i in range(len(sigmai)):
                fhc_expr = fhc_expr.subs(etas[i], variables[i][3])

            return fhc_expr

        fhc = _fhc_expression(sigmai, flag, densities)

    # -------------------------
    # Treat free energy as symbolic function
    # -------------------------
    variables = tuple(densities)
    F_hc = sp.Lambda(variables, fhc)

    result = {
        "variables": variables,
        "function": F_hc,
        "expression": fhc
    }

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json and ctx is not None:
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)

        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "sigma_eff": sigmai,
                    "flag": flag,
                    "variables": [str(v) for v in variables],
                    "function": str(F_hc),
                    "expression": str(fhc),
                },
                f,
                indent=2,
            )

        print(f"✅ Hard-core free energy exported: {out_file}")

    return result

