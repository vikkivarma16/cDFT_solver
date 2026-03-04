from sympy import Lambda


def free_energy_lattice(
    ctx=None,
    hc_data=None,
    export_json=True,
    filename="Solution_lattice.json",
):
    """
    Computes lattice free energy:

        F = F_hard_core (lattice / zero-d)
            + 1/2 sum_ij v_ij rho_i rho_j

    No squeezing. Pure quadratic mean-field.
    """

    import numpy as np
    import sympy as sp
    import json
    from pathlib import Path

    # -------------------------------------------------
    # Validate input
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
    # Extract diagonal helper
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

    sigma = extract_diagonal(sigma_raw, n_species, name="sigma")
    flag = extract_diagonal(flag_raw, n_species, name="flag")

    sigma = [float(s) for s in sigma]
    flag = [int(f) for f in flag]

    # -------------------------------------------------
    # Symbolic densities
    # -------------------------------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------------------------------
    # Hard-core lattice free energy
    # -------------------------------------------------
    if all(s == 0.0 for s in sigma) and all(f == 0 for f in flag):
        fhc = sp.Integer(0)
    else:
        etas = [sp.symbols(f"eta_{i}") for i in range(n_species)]

        # η_i = ρ_i * π σ_i^3 / 6
        eta_real = [
            densities[i] * (sp.pi * sigma[i] ** 3 / 6)
            for i in range(n_species)
        ]

        fac1 = 1 - sum(etas)
        fac2 = sum(etas[i] for i in range(n_species) if flag[i] == 1)

        phi0 = fac1 * sp.log(1 - fac2) + fac2

        fhc = phi0

        for i in range(n_species):
            fhc = fhc.subs(etas[i], eta_real[i])

    # -------------------------------------------------
    # Mean-field interaction symbols
    # -------------------------------------------------
    vij = [
        [sp.symbols(f"v_{species[i]}_{species[j]}")
         for j in range(n_species)]
        for i in range(n_species)
    ]

    # -------------------------------------------------
    # Mean-field free energy
    # -------------------------------------------------
    f_mf = sum(
        sp.Rational(1, 2)
        * vij[i][j]
        * densities[i]
        * densities[j]
        for i in range(n_species)
        for j in range(n_species)
    )

    # -------------------------------------------------
    # Total lattice free energy
    # -------------------------------------------------
    f_total = fhc + f_mf

    # -------------------------------------------------
    # Flatten variables for Lambda
    # -------------------------------------------------
    flat_vars = tuple(
        densities
        + [vij[i][j]
           for i in range(n_species)
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
                    "sigma_eff": sigma,
                    "flag": flag,
                    "variables": [str(v) for v in flat_vars],
                    "function": str(F_func),
                    "expression": str(f_total),
                },
                f,
                indent=4,
            )

        print(f"✅ Lattice free energy exported: {out_file}")

    return result
