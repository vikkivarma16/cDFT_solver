def free_energy_hard_core(ctx):
    """
    Computes the hard-core contribution to the free energy for a multi-species system.
    If no hard-core is present (all σ_eff=0, flag=0), returns fhc_symbolic = 0
    while keeping the same return format.
    """
    
    import numpy as np
    import sympy as sp
    from pathlib import Path
    import json
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials

    # -------------------------
    # Setup paths
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "Solution_hardcore.json"

    # -------------------------
    # Load hard-core data (sigma_eff and flag)
    # -------------------------
    hc_data = hard_core_potentials(ctx)

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hard_core_potentials returned invalid data")

    # Sort species alphabetically for reproducibility
    species = sorted(hc_data.keys())
    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]

    # -------------------------
    # Define symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{i}") for i in range(len(sigmai))]

    # -------------------------
    # Check if there is truly no hard-core
    # -------------------------
    if all(s == 0.0 for s in sigmai) and all(f == 0 for f in flag):
        fhc = sp.Integer(0)  # symbolic zero
    else:
        # -------------------------
        # Define hard-core free energy
        # -------------------------
        def _fhc_expression(sigmai, flag, densities):
            """Return symbolic expression for hard-core free energy."""
            measures = [[1, sigma/2, np.pi*sigma**2, sigma**3 * np.pi/6] for sigma in sigmai]
            etas = [sp.symbols(f"eta_{i}") for i in range(len(sigmai))]
            variables = [[densities[i] * measures[i][j] for j in range(4)] for i in range(len(sigmai))]

            fac1 = 1 - sum(etas)
            fac2 = sum(etas[i] for i in range(len(sigmai)) if flag[i] == 1)
            phi0 = fac1 * sp.log(1 - fac2) + fac2

            diff_1 = [sp.diff(phi0, etas[i]) for i in range(len(sigmai))]
            diff_2 = [[sp.diff(phi0, etas[i], etas[j]) for j in range(len(sigmai))] for i in range(len(sigmai))]
            diff_3 = [[[sp.diff(phi0, etas[i], etas[j], etas[k])
                        for k in range(len(sigmai))] for j in range(len(sigmai))]
                      for i in range(len(sigmai))]

            phi1 = sum(variables[i][0] * diff_1[i] for i in range(len(sigmai)))
            phi2 = sum(variables[i][1] * variables[j][2] * diff_2[i][j]
                       for i in range(len(sigmai)) for j in range(len(sigmai)))
            phi3 = sum((1/(24*np.pi)) * variables[i][2]*variables[j][2]*variables[k][2] * diff_3[i][j][k]
                       for i in range(len(sigmai)) for j in range(len(sigmai)) for k in range(len(sigmai)))

            fhc = phi1 + phi2 + phi3

            # Substitute η_i with actual expressions
            for i in range(len(sigmai)):
                fhc = fhc.subs(etas[i], variables[i][3])

            return fhc

        fhc = _fhc_expression(sigmai, flag, densities)

    # -------------------------
    # Prepare return data
    # -------------------------
    result = {
        "species": species,
        "sigma_eff": sigmai,
        "flag": flag,
        "densities_symbols": [str(d) for d in densities],
        "f_hc": fhc,
    }

    # -------------------------
    # Export results to JSON
    # -------------------------
    with open(output_file, "w") as f:
        json.dump(
            {
                "species": species,
                "sigma_eff": sigmai,
                "flag": flag,
                "densities_symbols": [str(d) for d in densities],
                "f_hc": str(fhc),
            },
            f,
            indent=4,
        )

    return result

