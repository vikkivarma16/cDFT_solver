import json
import numpy as np
import sympy as sp
from sympy import diff, log
from pathlib import Path
from scipy import integrate
from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials


def free_energy_SMF(ctx):
    """
    Computes the symbolic mean-field (SMF) free energy for a multi-species system.

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
    output_file = scratch / "Solution_SMF.json"

    # -------------------------
    # Load potentials and species
    # -------------------------
    potential_data = meanfield_potentials(ctx, mode="meanfield")
    species = potential_data["species"]
    interactions = potential_data["interactions"]

    nelement = len(species)
    if nelement == 0:
        raise ValueError("No species found in input data.")

    # -------------------------
    # Define densities and interaction symbols
    # -------------------------
    densities = [sp.symbols(f"rho_{i}") for i in range(nelement)]
    vij = [[sp.symbols(f"v_{i}_{j}") for j in range(nelement)] for i in range(nelement)]

    # -------------------------
    # Construct mean-field free energy
    # -------------------------
    f_mf = 0
    for i in range(nelement):
        for j in range(nelement):
            f_mf += sp.Rational(1, 2) * vij[i][j] * densities[i] * densities[j]

    # -------------------------
    # Substitute interaction parameters
    # (You can later map vij[i][j] from potentials if available)
    # -------------------------
    # Example: If needed, you can fill in vij[i][j] = interaction strength
    # from interactions["primary"][f"{species[i]}{species[j]}"]["epsilon"]

    # -------------------------
    # Create a numeric evaluator
    # -------------------------
    f_mf_func = sp.lambdify(densities, f_mf, "numpy")

    # -------------------------
    # Return both symbolic and functional forms
    # -------------------------
    result = {
    "species": species,
    "densities": [str(sym) for sym in densities],
    "vij": [[str(v) for v in row] for row in vij],
    "f_mf": f_mf,
}


    with open(output_file, "w") as f:
        json.dump({
        "species": species,
        "densities": [str(sym) for sym in densities],
        "vij": [[str(v) for v in row] for row in vij],
        "f_mf": str(f_mf)
    }, f, indent=4)

    return result



# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."
        input_file = "interactions.json"

    out = free_energy_SMF(Ctx())
    print("Species:", out["species"])
    print("Symbolic Free Energy:", out["free_energy_symbolic"])

