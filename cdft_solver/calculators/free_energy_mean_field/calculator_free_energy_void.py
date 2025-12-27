def free_energy_void(ctx):
    """
    Computes the symbolic cavity mean field (CMF) free energy for a multi-species system,
    including eta correction later supplied by the void particle correlation.
    needs voids-particles correlation data
    The function:
      - Loads mean-field potentials and hard-core parameters
      - Constructs the symbolic EMF free energy density
      - Returns all symbols and expressions for downstream processing

    Parameters
    ----------
    ctx : object
        Context with attributes:
            - scratch_dir : Path to scratch/output directory
            - input_file  : Path to the input JSON file

    Returns
    -------
    dict
        {
            "species": [...],
            "sigma_eff": [...],
            "flag": [...],
            "densities_symbols": [...],
            "interaction_symbols": [[...], [...]],
            "volume_factors": [...],
            "f_mf_symbolic": sympy.Expr
        }
    """
    import numpy as np
    import sympy as sp
    from pathlib import Path
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials

    # -------------------------
    # Setup paths
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "Solution_void.json"

    # -------------------------
    # Load data
    # -------------------------
    potential_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("Hard-core data could not be loaded or is invalid.")

    if potential_data is None or "species" not in potential_data:
        raise ValueError("Mean-field potential data is missing or invalid.")

    # -------------------------
    # Species and parameters
    # -------------------------
    species = sorted(hc_data.keys())  # ensure consistent order
    nelement = len(species)
    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]

    # -------------------------
    # Define symbols
    # -------------------------
    densities = [sp.symbols(f"rho_{i}") for i in range(len(sigmai))]
    vij = [[sp.symbols(f"v_{i}_{j}") for j in range(nelement)] for i in range(nelement)]

    # -------------------------
    # Compute volume correction factors
    # -------------------------
    volume_factor = []
    for j in range(nelement):
        factor = 0
        for i in range(nelement):
            if flag[i] == 1 and j != i:
                # average pair volume correction
                avg_p_vol = sigmai[i]  ** 3.0
                term =  (np.pi / 6) * avg_p_vol *densities[i]
                factor += term
        volume_factor.append(1 - factor)

    # -------------------------
    # Construct mean-field free energy (symbolic)
    # -------------------------
    f_mf = 0
    for i in range(nelement):
        for j in range(nelement):
            f_mf += sp.Rational(1, 2) * vij[i][j] * densities[i] * densities[j] / (volume_factor[j]*volume_factor[i])

    # -------------------------
    # solver results
    # -------------------------
    result = {
        "species": species,
        "sigma_eff": sigmai,
        "flag": flag,
        "densities": [str(d) for d in densities],
        "vij": [[str(vij[i][j]) for j in range(nelement)] for i in range(nelement)],
        "volume_factors": [str(vf) for vf in volume_factor],
        "f_mf": f_mf,
    }

    # -------------------------
    # (Optional) Export JSON for downstream use
    # -------------------------
    try:
        import json
        json_output = {
            "species": species,
            "sigma_eff": sigmai,
            "flag": flag,
            "densities": [str(d) for d in densities],
            "vij": [[str(vij[i][j]) for j in range(nelement)] for i in range(nelement)],
            "volume_factors": [sp.simplify(vf).__str__() for vf in volume_factor],
            "f_mf": sp.simplify(f_mf).__str__(),
        }
        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not export symbolic EMF free energy to JSON: {e}")

    return result


# Example standalone test
if __name__ == "__main__":
    class DummyCtx:
        scratch_dir = "."
        input_file = "interactions.json"

    out = free_energy_EMF(DummyCtx())
    print("Species:", out["species"])
    print("σ_eff:", out["sigma_eff"])
    print("Flags:", out["flag"])
    print("Densities:", out["densities_symbols"])
    print("Interactions:", out["interaction_symbols"])
    print("Free energy symbolic expression:\n", out["f_mf_symbolic"])

