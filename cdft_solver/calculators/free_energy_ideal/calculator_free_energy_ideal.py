import json
import numpy as np
import sympy as sp
from sympy import log
from pathlib import Path
from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials


def free_energy_ideal(ctx):
    """
    Computes the symbolic ideal (entropic) part of the free energy for a multi-species system.

    Parameters
    ----------
    ctx : object
        Context with attributes:
            - scratch_dir : str or Path → working directory
            - input_file  : str or Path → path to JSON defining interactions

    Returns
    -------
    dict
        {
            "species": [...],
            "free_energy_symbolic": sympy.Expr,
            "free_energy_func": callable (numeric evaluator)
        }
    """

    # -------------------------
    # Setup paths
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "Solution_ideal.json"

    # -------------------------
    # Load species list (reuse same potential loader for consistency)
    # -------------------------
    potential_data = meanfield_potentials(ctx, mode="meanfield")
    species = potential_data["species"]

    nelement = len(species)
    if nelement == 0:
        raise ValueError("No species found in input data.")

    # -------------------------
    # Define density symbols
    # -------------------------
    densities = [sp.symbols(f"rho_{i}") for i in range(nelement)]

    # -------------------------
    # Construct ideal free energy
    # f_ideal = sum_i [ rho_i * (log(rho_i) - 1) ]
    # -------------------------
    f_ideal = sum(rho * (sp.log(rho) - 1) for rho in densities)

    # -------------------------
    # Create a numeric evaluator
    # -------------------------
    f_ideal_func = sp.lambdify(densities, f_ideal, "numpy")

    # -------------------------
    # Save to JSON
    # -------------------------
    with open(output_file, "w") as f:
        json.dump({
            "species": species,
            "densities": [str(sym) for sym in densities],
            "f_ideal": str(f_ideal)
        }, f, indent=4)

    # -------------------------
    # Return both symbolic and functional forms
    # -------------------------
    result = {
        "species": species,
        "densities": [str(sym) for sym in densities],
        "f_ideal": f_ideal
    }

    return result



# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."
        input_file = "interactions.json"

    out = free_energy_ideal(Ctx())
    print("Species:", out["species"])
    print("Symbolic Ideal Free Energy:", out["f_ideal"])

