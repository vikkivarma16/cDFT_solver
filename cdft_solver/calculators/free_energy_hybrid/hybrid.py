from sympy import Lambda


def free_energy_hybrid(ctx=None, export_json=True, filename="Solution_hybrid.json"):
    """
    Computes the symbolic hybrid free energy:

        F_hybrid = F_hard_core (FMT-like)
                   + squeezed mean-field interaction

    Interaction structure:
        - Colloid–colloid: 1/2 v_ij η_i η_j
        - Polymer–polymer: 1/2 v_ij η_i η_j / (1 - η_c)
        - Polymer–colloid: 1/2 v_ij η_i η_j / (1 - η_c)

    where:
        η_c = sum over colloid packing fractions

    Parameters
    ----------
    ctx : object
        Must contain:
            - scratch_dir
            - input_file

    Returns
    -------
    dict with symbolic function and expression
    """

    import sympy as sp
    import numpy as np
    import json
    from pathlib import Path
    from cdft_solver.generators.potential_splitters.generator_potential_splitter_hc import (
        hard_core_potentials,
    )

    print("Hybrid cavity + squeezed mean-field free energy constructed.\n")

    # -------------------------------------------------
    # Load hard-core data
    # -------------------------------------------------
    hc_data = hard_core_potentials(ctx)

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("Hard-core data could not be loaded.")

    species = sorted(hc_data.keys())
    n_species = len(species)

    sigma = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]  # 1 = colloid, 0 = polymer

    # -------------------------------------------------
    # Symbolic densities
    # -------------------------------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------------------------------
    # Packing fractions η_i
    # -------------------------------------------------
    eta = [
        densities[i] * (sp.pi * sigma[i] ** 3 / 6)
        for i in range(n_species)
    ]

    # Colloid packing fraction
    eta_c = sum(
        eta[i] for i in range(n_species) if flag[i] == 1
    )

    # -------------------------------------------------
    # Hard-core FMT part (zero-dimensional route)
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
            etas_sym[i] for i in range(n_species) if flag[i] == 1
        )

        phi0 = fac1 * sp.log(1 - fac2) + fac2

        diff1 = [sp.diff(phi0, etas_sym[i]) for i in range(n_species)]
        diff2 = [
            [sp.diff(phi0, etas_sym[i], etas_sym[j])
             for j in range(n_species)]
            for i in range(n_species)
        ]
        diff3 = [
            [
                [sp.diff(phi0, etas_sym[i], etas_sym[j], etas_sym[k])
                 for k in range(n_species)]
                for j in range(n_species)
            ]
            for i in range(n_species)
        ]

        phi1 = sum(
            variables[i][0] * diff1[i]
            for i in range(n_species)
        )

        phi2 = sum(
            variables[i][1] * variables[j][2] * diff2[i][j]
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

        # Substitute η_i
        for i in range(n_species):
            fhc = fhc.subs(etas_sym[i], variables[i][3])

        return fhc

    fhc = hard_core_expression()

    # -------------------------------------------------
    # Mean-field interaction symbols
    # -------------------------------------------------
    vij = [
        [sp.symbols(f"v_{species[i]}_{species[j]}")
         for j in range(n_species)]
        for i in range(n_species)
    ]

    # -------------------------------------------------
    # Squeezed interaction free energy
    # -------------------------------------------------
    f_mf = sp.Integer(0)

    for i in range(n_species):
        for j in range(i, n_species):

            pref = 1 if i == j else 2

            # Polymer involved?
            if flag[i] == 0 or flag[j] == 0:
                squeeze = 1 / (1 - eta_c)
            else:
                squeeze = 1

            f_mf += (
                sp.Rational(1, 2)
                * pref
                * vij[i][j]
                * eta[i]
                * eta[j]
                * squeeze
            )

    # -------------------------------------------------
    # Total hybrid free energy
    # -------------------------------------------------
    f_total = fhc + f_mf

    # -------------------------------------------------
    # Flatten variables for Lambda
    # -------------------------------------------------
    flat_vars = tuple(
        densities
        + [vij[i][j] for i in range(n_species)
           for j in range(n_species)]
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
                    "variables": [str(v) for v in flat_vars],
                    "function": str(F_func),
                    "expression": str(f_total),
                },
                f,
                indent=4,
            )

        print(f"✅ Hybrid free energy exported: {out_file}")

    return result
