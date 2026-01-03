def hybrid_planer(ctx):
    """
    Computes the hard-core contribution to the free energy for a multi-species system.

    The function:
      - Loads hard-core consistency data from JSON via `hard_core_data(ctx)`
      - Builds a symbolic expression for the hard-core free energy (fhc)
      - Returns fhc and associated symbolic variables for downstream use

    Parameters
    ----------
    ctx : object
        Context with attributes:
            - scratch_dir : Path to scratch directory
            - input_file  : Path to the input JSON file

    Returns
    -------
    dict
        {
            "species": [...],
            "sigma_eff": [...],
            "flag": [...],
            "densities_symbols": [...],
            "fhc_symbolic": sympy.Expr
        }
    """
    import numpy as np
    import sympy as sp
    from pathlib import Path
    from cdft_solver.generators.potential_splitters.generator_potential_splitter_hc import hard_core_potentials

    # -------------------------
    # Setup paths
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "Solution.json"

    # -------------------------
    # Load hard-core data (sigma_eff and flag)
    # -------------------------
    hc_data = hard_core_potentials(ctx)

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("Hard-core data could not be loaded or is invalid.")

    # Sort species alphabetically for reproducibility
    species = sorted(hc_data.keys())

    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]

    # -------------------------
    # Define symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # Define hard-core free energy
    # -------------------------
    def _fhc_expression(sigmai, flag, densities):
        """Return symbolic expression for hard-core free energy."""
        # Geometrical measures: m0, m1, m2, m3
        measures = [[1, sigma/2, np.pi*sigma**2, sigma**3 * np.pi/6] for sigma in sigmai]

        # Define packing fractions η_i
        etas = [sp.symbols(f"eta_{i}") for i in range(len(sigmai))]

        # Map densities → η_i via measures
        variables = [[densities[i] * measures[i][j] for j in range(4)] for i in range(len(sigmai))]

        # Build φ0
        fac1 = 1 - sum(etas)
        fac2 = sum(etas[i] for i in range(len(sigmai)) if flag[i] == 1)
        phi0 = fac1 * sp.log(1 - fac2) + fac2

        # Compute derivatives
        diff_1 = [sp.diff(phi0, etas[i]) for i in range(len(sigmai))]
        diff_2 = [[sp.diff(phi0, etas[i], etas[j]) for j in range(len(sigmai))] for i in range(len(sigmai))]
        diff_3 = [[[sp.diff(phi0, etas[i], etas[j], etas[k])
                    for k in range(len(sigmai))] for j in range(len(sigmai))]
                  for i in range(len(sigmai))]

        # Compute contributions
        phi1 = sum(variables[i][0] * diff_1[i] for i in range(len(sigmai)))
        phi2 = sum(variables[i][1] * variables[j][2] * diff_2[i][j]
                   for i in range(len(sigmai)) for j in range(len(sigmai)))
        phi3 = sum((1/(24*np.pi)) * variables[i][2]*variables[j][2]*variables[k][2] * diff_3[i][j][k]
                   for i in range(len(sigmai)) for j in range(len(sigmai)) for k in range(len(sigmai)))

        # Total free energy density
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
        "fhc_symbolic": fhc,
    }

    return result


# Example standalone test
if __name__ == "__main__":
    class DummyCtx:
        scratch_dir = "."
        input_file = "interactions.json"

    out = free_energy_hard_core(DummyCtx())
    print("Species:", out["species"])
    print("σ_eff:", out["sigma_eff"])
    print("Flags:", out["flag"])
    print("Densities:", out["densities_symbols"])
    print("Free energy symbolic expression:\n", out["fhc_symbolic"])

