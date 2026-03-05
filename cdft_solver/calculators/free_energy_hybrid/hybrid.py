from sympy import Lambda


def hybrid(
    ctx=None,
    hc_data=None,
    export_json=True,
    filename="Solution_hybrid.json",
):
    """
    Computes hybrid free energy:

        F = F_hard_core (FMT-like zero-d)
            + squeezed mean-field interaction

    Interaction structure:
        - Colloid–colloid: 1/2 v_ij η_i η_j
        - Polymer–polymer: 1/2 v_ij η_i η_j / (1 - η_c)
        - Polymer–colloid: 1/2 v_ij η_i η_j / (1 - η_c)

    where:
        η_c = total colloid packing fraction
    """

    import sympy as sp
    import numpy as np
    import json
    from pathlib import Path

    # -------------------------------------------------
    # Validate hc_data
    # -------------------------------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

    if not species:
        raise ValueError("hc_data must contain 'species'")

    n_species = len(species)

    # -------------------------------------------------
    # Extract diagonal helper (same as lattice)
    # -------------------------------------------------
    def extract_diagonal(data, n, name="array"):

        arr = np.asarray(data)

        if arr.ndim == 1 and arr.size == n:
            return arr.tolist()

        if arr.ndim == 1 and arr.size == n * n:
            return arr.reshape((n, n)).diagonal().tolist()

        if arr.ndim == 2 and arr.shape == (n, n):
            return arr.diagonal().tolist()

        raise ValueError(
            f"{name} must be length-{n}, {n}x{n}, or flat length-{n*n} array"
        )

    sigma = extract_diagonal(sigma_raw, n_species, "sigma")
    flag = extract_diagonal(flag_raw, n_species, "flag")

    sigma = [float(s) for s in sigma]
    flag = [int(f) for f in flag]

    # -------------------------------------------------
    # Symbolic densities
    # -------------------------------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------------------------------
    # Packing fractions
    # -------------------------------------------------
    eta = [
        densities[i] * (sp.pi * sigma[i] ** 3 / 6)
        for i in range(n_species)
    ]

    # total colloid packing fraction
    
    
     # -------------------------------------------------
    # Mean-field interaction symbols
    # -------------------------------------------------
    vij = [
        [
            sp.symbols(f"v_{species[i]}_{species[j]}")
            for j in range(n_species)
        ]
        for i in range(n_species)
    ]

    # -------------------------------------------------
    # Squeezed mean-field interaction
    # -------------------------------------------------
    f_mf = sp.Integer(0)
    
    etas_sym = [sp.symbols(f"eta_{i}") for i in range(n_species)]
    eta_c = sum(
        etas_sym[i] for i in range(n_species)
        if flag[i] == 1
    )

    for i in range(n_species):
        for j in range(n_species):

            if flag[i] == 0 or flag[j] == 0:
                squeeze = 1 / (1 - eta_c)
            else:
                squeeze = 1

            f_mf += (
                sp.Rational(1, 2)
                * vij[i][j]
                * etas_sym[i]
                * etas_sym[j]
                * squeeze
            )
    
    

    # -------------------------------------------------
    # Hard-core FMT-like term
    # -------------------------------------------------
    def hard_core_expression():

        measures = [
            [1, sigma[i] / 2, sp.pi * sigma[i] ** 2, sp.pi * sigma[i] ** 3 / 6]
            for i in range(n_species)
        ]

        etas_sym = [sp.symbols(f"eta_{i}") for i in range(n_species)]

        variables = [
            [densities[i] * measures[i][j] for j in range(4)]
            for i in range(n_species)
        ]

        fac1 = 1 - sum(etas_sym)

        fac2 = sum(
            etas_sym[i]
            for i in range(n_species)
            if flag[i] == 1
        )

        phi0 = fac1 * sp.log(1 - fac2) + fac2 + f_mf

        diff1 = [
            sp.diff(phi0, etas_sym[i])
            for i in range(n_species)
        ]
        
        #print ("\n\n\n\n", diff1, "\n\n\n\n")
        
        diff2 = [
            [
                sp.diff(phi0, etas_sym[i], etas_sym[j])
                for j in range(n_species)
            ]
            for i in range(n_species)
        ]
        
        #print ("\n\n\n\n", diff2, "\n\n\n\n")

        diff3 = [
            [
                [
                    sp.diff(phi0, etas_sym[i], etas_sym[j], etas_sym[k])
                    for k in range(n_species)
                ]
                for j in range(n_species)
            ]
            for i in range(n_species)
        ]
        
        #print ("\n\n\n\n", diff3, "\n\n\n\n")

        phi1 = sum(
            variables[i][0] * diff1[i]
            for i in range(n_species)
        )

        phi2 = sum(
            variables[i][1]
            * variables[j][2]
            * diff2[i][j]
            for i in range(n_species)
            for j in range(n_species)
        )

        phi3 = sum(
            (1 / (24 * sp.pi))
            * variables[i][2]
            * variables[j][2]
            * variables[k][2]
            * diff3[i][j][k]
            for i in range(n_species)
            for j in range(n_species)
            for k in range(n_species)
        )

        fhc = phi1 + phi2 + phi3

        # substitute η_i
        for i in range(n_species):
            fhc = fhc.subs(etas_sym[i], variables[i][3])

        return fhc

    fhc = hard_core_expression()

   

    # -------------------------------------------------
    # Total free energy
    # -------------------------------------------------
    f_total = fhc

    # -------------------------------------------------
    # Flatten variables
    # -------------------------------------------------
    flat_vars = tuple(
        densities
        + [
            vij[i][j]
            for i in range(n_species)
            for j in range(n_species)
        ]
    )

    F_func = Lambda(flat_vars, f_total)

    result = {
        "species": species,
        "sigma_eff": sigma,
        "flag": flag,
        "variables": flat_vars,
        "function": F_func,
        "expression": f_total,
    }

    # -------------------------------------------------
    # Optional JSON export
    # -------------------------------------------------
    if export_json and ctx is not None and hasattr(ctx, "scratch_dir"):

        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)

        out_file = scratch / filename

        with open(out_file, "w") as f:

            json.dump(
                {
                    "species": species,
                    "sigma_eff": sigma,
                    "flag": flag,
                    "variables": [str(v) for v in flat_vars],
                    "function": str(F_func),
                    "expression": str(f_total),
                },
                f,
                indent=4,
            )

        print(f"✅ Hybrid free energy exported: {out_file}")

    return result
