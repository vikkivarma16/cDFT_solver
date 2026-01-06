def hard_core_planer(
    ctx=None,
    hc_data=None,
    export_json=True,
    filename="Solution_hardcore_z.json"
):
    """
    Computes the hard-core free energy (FMT-like) for multi-species system
    using n_j weight densities (n0..n5) and η = n3_i.

    Parameters
    ----------
    ctx : object, optional
        Must have `scratch_dir` for exporting JSON.
    hc_data : dict
        {
            "species": [...],
            "sigma": [...] or n×n,
            "flag": [...] or n×n,
            "potentials": {...}  # ignored here
        }
    export_json : bool
        Export symbolic result to JSON
    filename : str
        JSON filename if export_json is True

    Returns
    -------
    dict
        {
            "species": [...],
            "sigma_eff": [...],
            "flag": [...],
            "variables": [[sympy.Symbol]*6 per species],
            "etas": [sympy.Symbol per species],
            "function": sympy.Lambda,
            "phi_total": sympy.Expr
        }
    """

    import sympy as sp
    import numpy as np
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
    # Define symbolic weight densities n0..n5
    # -------------------------
    variables = [
        [sp.symbols(f"n{j}_{s}") for j in range(6)]
        for s in species
    ]
    etas = [variables[i][3] for i in range(n_species)]

    # -------------------------
    # Shortcut for no hard-core
    # -------------------------
    if all(s == 0.0 for s in sigmai) and all(f == 0 for f in flag):
        total_phi = sp.Integer(0)
    else:
        # -------------------------
        # φ0
        # -------------------------
        fac1 = 1 - sum(etas)
        fac2 = sum(etas[i] for i in range(n_species) if flag[i] == 1)
        phi0 = fac1 * sp.log(1 - fac2) + fac2

        # -------------------------
        # Derivatives
        # -------------------------
        diff_1 = [sp.diff(phi0, etas[i]) for i in range(n_species)]
        diff_2 = [[sp.diff(phi0, etas[i], etas[j]) for j in range(n_species)] for i in range(n_species)]
        diff_3 = [[[sp.diff(phi0, etas[i], etas[j], etas[k]) for k in range(n_species)] for j in range(n_species)] for i in range(n_species)]

        # -------------------------
        # φ1, φ2, φ3
        # -------------------------
    
        
        phi1 = sum(variables[i][0] * diff_1[i] for i in range(n_species))
        phi2 = sum((variables[i][1] * variables[j][2] - variables[i][4] * variables[j][5]) * diff_2[i][j] for i in range(n_species) for j in range(n_species))
        phi3 = (1/(8*np.pi)) *sum( ((variables[i][2]*variables[j][2]*variables[k][2]/3.0)  - variables[i][2] *variables[j][5] * variables[k][5] + (3.0/2.0) * ( variables[i][5] *variables[k][5] * ((variables[j][2] - 4.0 * variables[j][3]/sigmai[j]) - variables[j][2]/3 ) - ((variables[i][2] - 4.0 * variables[i][3]/sigmai[i]) - variables[i][2]/3) * ((variables[j][2] - 4 * variables[j][3]/sigmai[j]) - variables[j][2]/3) * ((variables[k][2] - 4 * variables[k][3]/sigmai[k]) - variables[k][2]/3 ) + 2.0 * ( ((variables[i][2] - 4.0 * variables[i][3]/sigmai[i]) - variables[i][2]/3 )/2 *((variables[j][2] - 4 * variables[j][3]/sigmai[j]) - variables[j][2]/3 )/2 *((variables[k][2] - 4 * variables[k][3]/sigmai[k]) - variables[k][2]/3 )/2 )    )  ) * diff_3[i][j][k] for i in range(n_species) for j in range(n_species) for k in range(n_species) )
            
        
        
        total_phi = phi1 + phi2 + phi3

    # -------------------------
    # Treat as symbolic function
    # -------------------------
    flat_variables = [v for var_list in variables for v in var_list]
    F_hc = sp.Lambda(tuple(flat_variables), total_phi)

    # -------------------------
    # Prepare result
    # -------------------------
    result = {
        
        "variables": variables,  # nested list n_species x 6
        "function": F_hc,
        "expression": total_phi
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
                    "variables": [[str(v) for v in var_list] for var_list in variables],
                    "function": str(F_hc),
                    "expression": str(total_phi)
                },
                f,
                indent=4
            )
        print(f"✅ Hard-core free energy exported: {out_file}")

    return result

