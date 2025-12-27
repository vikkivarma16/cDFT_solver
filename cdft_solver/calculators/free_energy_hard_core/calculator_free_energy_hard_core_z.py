def free_energy_hard_core_z(ctx):
    """
    Computes the hard-core contribution to the free energy for a multi-species system,
    now using the updated:
        - n_j weight density variables
        - eta = n3_i
        - updated phi0, phi1, phi2, phi3 expressions
        - total_phi exported
    """

    import numpy as np
    import sympy as sp
    from pathlib import Path
    import json
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials

    # ------------------------------------------------
    # Setup scratch / I/O
    # ------------------------------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "Solution_hardcore.json"

    # ------------------------------------------------
    # Load hard-core parameters
    # ------------------------------------------------
    hc_data = hard_core_potentials(ctx)

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hard_core_potentials returned invalid data")

    species = sorted(hc_data.keys())
    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]

    nspecies = len(species)

    # ------------------------------------------------
    # Define symbolic weight densities n0...n5
    # variables[i][j] = n_j_(i), j=0..5
    # ------------------------------------------------
    variables = [
        [sp.symbols(f"n{j}_{i}") for j in range(6)]
        for i in range(nspecies)
    ]

    # eta = n_3 (third weight density)
    etas = [variables[i][3] for i in range(nspecies)]

    # ------------------------------------------------
    # If no hard-core effect is present
    # ------------------------------------------------
    if all(s == 0.0 for s in sigmai) and all(f == 0 for f in flag):
        total_phi = sp.Integer(0)

    else:
        # ------------------------------------------------
        # Construct φ0
        # ------------------------------------------------
        fac1 = 1 - sum(etas)
        fac2 = sum(etas[i] for i in range(nspecies) if flag[i] == 1)

        phi0 = fac1 * sp.log(1 - fac2) + fac2

        # ------------------------------------------------
        # Compute derivatives wrt etas
        # ------------------------------------------------
        diff_1 = [sp.diff(phi0, etas[i]) for i in range(nspecies)]
        diff_2 = [
            [sp.diff(phi0, etas[i], etas[j]) for j in range(nspecies)]
            for i in range(nspecies)
        ]
        diff_3 = [
            [
                [sp.diff(phi0, etas[i], etas[j], etas[k]) for k in range(nspecies)]
                for j in range(nspecies)
            ]
            for i in range(nspecies)
        ]

        # ------------------------------------------------
        # φ1
        # ------------------------------------------------
        phi1 = sum(
            variables[i][0] * diff_1[i]
            for i in range(nspecies)
        )

        # ------------------------------------------------
        # φ2
        # (n1_i n2_j  – n4_i n5_j) term
        # ------------------------------------------------
        phi2 = sum(
            (variables[i][1] * variables[j][2] -
             variables[i][4] * variables[j][5]) * diff_2[i][j]
            for i in range(nspecies)
            for j in range(nspecies)
        )

        # ------------------------------------------------
        # φ3 — updated long expression
        # ------------------------------------------------
        phi3 = (1/(8*np.pi)) * sum(
            (
                (variables[i][2]*variables[j][2]*variables[k][2] / 3.0)
                - variables[i][2] * variables[j][5] * variables[k][5]
                + (3.0/2.0) * (
                    variables[i][5] * variables[k][5] *
                    ((variables[j][2] - 4.0*variables[j][3]/sigmai[j]) - variables[j][2]/3)
                    - ((variables[i][2] - 4.0*variables[i][3]/sigmai[i]) - variables[i][2]/3)
                    * ((variables[j][2] - 4.0*variables[j][3]/sigmai[j]) - variables[j][2]/3)
                    * ((variables[k][2] - 4.0*variables[k][3]/sigmai[k]) - variables[k][2]/3)
                    + 2.0 * (
                        ((variables[i][2] - 4.0*variables[i][3]/sigmai[i]) - variables[i][2]/3)/2
                        * ((variables[j][2] - 4.0*variables[j][3]/sigmai[j]) - variables[j][2]/3)/2
                        * ((variables[k][2] - 4.0*variables[k][3]/sigmai[k]) - variables[k][2]/3)/2
                    )
                )
            ) * diff_3[i][j][k]
            for i in range(nspecies)
            for j in range(nspecies)
            for k in range(nspecies)
        )

        phi3 = sp.simplify(phi3)

        # ------------------------------------------------
        # Total free-energy density
        # ------------------------------------------------
        total_phi = phi1 + phi2 + phi3

    # ------------------------------------------------
    # Export Variables
    # ------------------------------------------------
    # flatten variable names for export
    variable_names = [
        [str(variables[i][j]) for j in range(6)]
        for i in range(nspecies)
    ]

    result = {
        "species": species,
        "sigma_eff": sigmai,
        "flag": flag,
        "variables": variable_names,
        "etas": [str(e) for e in etas],
        "phi_total": total_phi,
    }

    # export JSON
    with open(output_file, "w") as f:
        json.dump(
            {
                "species": species,
                "sigma_eff": sigmai,
                "flag": flag,
                "variables": variable_names,
                "etas": [str(e) for e in etas],
                "phi_total": str(total_phi),
            },
            f,
            indent=4,
        )

    return result

