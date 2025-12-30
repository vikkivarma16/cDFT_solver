import json
import numpy as np
import sympy as sp
from pathlib import Path

def free_energy_SMF(ctx=None, hc_data=None, export_json=True, filename="Solution_SMF.json"):
    """
    Computes the symbolic mean-field (SMF) free energy for a multi-species system
    without hard-core corrections.

    Parameters
    ----------
    ctx : object, optional
        Must have `scratch_dir` for exporting JSON.
    hc_data : dict
        Species information, e.g.,
        {
            "species": [...],
            "sigma": [...],  # optional
            "flag": [...]    # optional
        }
    export_json : bool
        Whether to save symbolic results to JSON.
    filename : str
        JSON output filename.

    Returns
    -------
    dict
        {
            "species": [...],
            "densities": [...],
            "vij": [[...], [...]],
            "variables": tuple,
            "f_mf_func": sympy.Lambda,
            "f_mf": sympy.Expr
        }
    """

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary with 'species'")

    species = list(hc_data.get("species", []))
    n_species = len(species)
    if n_species == 0:
        raise ValueError("No species provided in hc_data.")

    # -------------------------
    # Symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # Symbolic interaction symbols
    # -------------------------
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
           for i in range(n_species)]

    # -------------------------
    # Construct symbolic SMF free energy
    # -------------------------
    f_mf = sum(sp.Rational(1, 2) * vij[i][j] * densities[i] * densities[j]
               for i in range(n_species) for j in range(n_species))

    # -------------------------
    # Flatten all variables into 1D tuple for Lambda
    # -------------------------
    flat_vars = tuple(densities + [vij[i][j] for i in range(n_species) for j in range(n_species)])
    f_mf_func = sp.Lambda(flat_vars, f_mf)

    # -------------------------
    # Prepare result
    # -------------------------
    result = {
        "variables": flat_vars,
        "function": f_mf_func,
        "expression": f_mf,
    }

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json and ctx is not None and hasattr(ctx, "scratch_dir"):
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump({
                "species": species,
                "variables": [str(v) for v in flat_vars],
                "function": str(f_mf_func),
                "expression": str(f_mf),
            }, f, indent=4)
        print(f"✅ SMF free energy exported: {out_file}")

    return result


# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."
    
    hc_data_example = {
        "species": ["A", "B"]
    }

    out = free_energy_SMF(Ctx(), hc_data=hc_data_example)
    print("Species:", out["species"])
    print("Densities:", out["densities"])
    print("Flattened variables tuple:", out["variables"])
    print("Symbolic SMF free energy:\n", out["f_mf"])

