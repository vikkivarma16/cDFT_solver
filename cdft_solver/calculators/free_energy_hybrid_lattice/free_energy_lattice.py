
from cdft_solver.calculators.free_energy_hard_core.hard_core_lattice import hard_core_lattice


def free_energy_lattice(ctx=None, hc_data=None, export_json=True, filename="Solution_lattice.json"):
    """
    Computes the symbolic lattice free energy by fusing:
      - hard-core free energy (FMT / zero-d limit)
      - mean-field quadratic interactions

    F_lattice = F_hc + 1/2 * sum_{ij} v_ij * rho_i * rho_j
    """

    import sympy as sp
    import json
    from pathlib import Path

    # -------------------------
    # Validate input
    # -------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    if not species:
        raise ValueError("hc_data must contain 'species'")

    n_species = len(species)

    # -------------------------
    # Symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # Hard-core contribution
    # -------------------------
    # Reuse your hard_core machinery conceptually:
    # same flags, same sigma handling, same zero-d construction
    from sympy import Lambda

    # Build hard-core free energy expression
    # (this is exactly what hard_core() already returns)
    hc_result = hard_core_lattice( ctx=None, hc_data=hc_data, export_json=False )

    fhc = hc_result["expression"]

    # -------------------------
    # Mean-field interaction symbols
    # -------------------------
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}")
            for j in range(n_species)]
           for i in range(n_species)]

    # -------------------------
    # Mean-field free energy
    # -------------------------
    f_mf = sum(
        sp.Rational(1, 2) * vij[i][j] * densities[i] * densities[j]
        for i in range(n_species)
        for j in range(n_species)
    )

    # -------------------------
    # Total lattice free energy
    # -------------------------
    f_lattice = fhc + f_mf

    # -------------------------
    # Flatten variables for Lambda
    # -------------------------
    flat_vars = tuple(
        densities
        + [vij[i][j] for i in range(n_species) for j in range(n_species)]
    )

    F_lattice_func = Lambda(flat_vars, f_lattice)

    result = {
        "variables": flat_vars,
        "function": F_lattice_func,
        "expression": f_lattice,
    }

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json and ctx is not None and hasattr(ctx, "scratch_dir"):
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)

        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "variables": [str(v) for v in flat_vars],
                    "function": str(F_lattice_func),
                    "expression": str(f_lattice),
                },
                f,
                indent=4,
            )

        print(f"✅ Lattice free energy exported: {out_file}")

    return result

