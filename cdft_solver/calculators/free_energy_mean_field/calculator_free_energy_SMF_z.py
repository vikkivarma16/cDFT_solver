import sympy as sp
import json
from pathlib import Path

def free_energy_SMF_z(ctx=None, hc_data=None, export_json=True, filename="Solution_SMF_z.json"):
    """
    Standard Mean-Field (SMF) two-point free-energy kernel:

        f(z, zs) = 1/2 ∑_{i,j} rho_i(z) * v_ij * rho_j(zs)

    Notes
    -----
    • No volume-factor or hard-core corrections
    • Fully symbolic two-point kernel
    • No spatial integration performed
    """

    # -------------------------
    # Validate input
    # -------------------------
     if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    n_species = len(species)

    if n_species == 0:
        raise ValueError("species list is empty")

    # -------------------------
    # Spatial coordinates
    # -------------------------
    z, zs = sp.symbols("z zs", real=True)

    # -------------------------
    # Two-point density symbols
    # -------------------------
    rho_z  = [sp.symbols(f"rho_{s}_z")  for s in species]
    rho_zs = [sp.symbols(f"rho_{s}_zs") for s in species]

    # -------------------------
    # Pair interaction symbols
    # -------------------------
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}")
            for j in range(n_species)]
            for i in range(n_species)]

    # -------------------------
    # SMF two-point kernel
    # -------------------------
    f_smf_z = 0
    for i in range(n_species):
        for j in range(n_species):
            f_smf_z += (
                sp.Rational(1, 2)
                * rho_z[i]
                * vij[i][j]
                * rho_zs[j]
            )

    # -------------------------
    # Flatten variables
    # -------------------------
    flat_vars = tuple(
        rho_z +
        rho_zs +
        [vij[i][j] for i in range(n_species) for j in range(n_species)]
    )

    f_smf_z_func = sp.Lambda(flat_vars, f_smf_z)

    # -------------------------
    # Result dictionary
    # -------------------------
    result = {
        "variables": flat_vars,
        "function": f_smf_z_func,
        "expression": f_smf_z,
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
                    "function": str(f_smf_z_func),
                    "expression": str(f_smf_z),
                },
                f,
                indent=4,
            )

        print(f"✅ SMF(z,zs) kernel exported: {out_file}")

    return result

