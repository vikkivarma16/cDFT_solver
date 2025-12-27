# Two-point SMF free-energy generator
# Analogous to EMF two-point version but without volume-factor corrections.

import sympy as sp
import numpy as np
from pathlib import Path

def free_energy_SMF_z(ctx):
    """
    Standard Mean-Field (SMF) free-energy functional in two-point form:

        F = 1/2 ∑_{i,j} ∫ dz ∫ dzs  densities_i(z)  V_ij(|z - zs|)  densities_j(zs)

    This version introduces two spatial arguments z and zs so that the
    free-energy kernel is explicitly a two-point functional.
    """

    from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials

    # -------------------------
    # Setup
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load interaction data
    # -------------------------
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    species = pot_data["species"]
    nelement = len(species)

    if nelement == 0:
        raise ValueError("No species detected in potential file.")

    # -------------------------
    # Define two-point density fields
    # -------------------------

    densities_z  = [sp.symbols(f"densities_z_{i}")  for i in range(nelement)]
    densities_zs = [sp.symbols(f"densities_zs_{i}") for i in range(nelement)]

    # -------------------------
    # Define interaction kernels V_ij(|z - zs|)
    # -------------------------
    r = sp.Abs(z - zs)
    vij = [[sp.symbols(f"v_{i}_{j}") for j in range(nelement)] for i in range(nelement)]

    # -------------------------
    # Construct two-point SMF free-energy density
    # -------------------------
    f_mf = 0
    for i in range(nelement):
        for j in range(nelement):
            f_mf += sp.Rational(1,2) * densities_z[i] * vij[i][j] * densities_zs[j]

    # -------------------------
    # Return symbolic structure
    # -------------------------
    return {
        "species": species,
        "densities_z": densities_z,
        "densities_zs": densities_zs,
        "vij": vij,
        "f_mf_two_point": f_mf,
    }


