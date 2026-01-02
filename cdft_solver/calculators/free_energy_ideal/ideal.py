import json
import sympy as sp
from pathlib import Path

def ideal(ctx=None, hc_data=None, export_json=True, filename="Solution_ideal.json"):
    """
    Computes the symbolic ideal (entropic) part of the free energy for a multi-species system.

    Parameters
    ----------
    ctx : object, optional
        Must have `scratch_dir` for exporting JSON.
    hc_data : dict
        {
            "species": [...],   # list of species names
            "potentials": {...} # ignored here
        }
    export_json : bool
        If True, export symbolic result to JSON
    filename : str
        JSON output filename

    Returns
    -------
    dict
        {
            "species": [...],
            "variables": [sympy.Symbol, ...],
            "function": sympy.Lambda,
            "expression": sympy.Expr
        }
    """

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary with 'species' key")

    species = list(hc_data.get("species", []))
    if not species:
        raise ValueError("No species found in hc_data")

    # -------------------------
    # Define symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # Construct ideal free energy
    # -------------------------
    f_ideal_expr = sum(rho * (sp.log(rho) - 1) for rho in densities)

    # -------------------------
    # Treat as symbolic function
    # -------------------------
    f_ideal_func = sp.Lambda(tuple(densities), f_ideal_expr)

    # -------------------------
    # Prepare result
    # -------------------------
    result = {
        "variables": densities,
        "function": f_ideal_func,
        "expression": f_ideal_expr
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
                "variables": [str(v) for v in densities],
                "function": str(f_ideal_func),
                "expression": str(f_ideal_expr)
            }, f, indent=4)

        print(f"✅ Ideal free energy exported: {out_file}")

    return result


# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."

    hc_data_example = {
        "species": ["A", "B", "C"],
        "potentials": {}  # ignored here
    }

    out = ideal(Ctx(), hc_data=hc_data_example)
    print("Species:", out["species"])
    print("Variables:", out["variables"])
    print("Symbolic Ideal Free Energy:", out["expression"])

