# Updated EMF free-energy generator
# Now written using two-point densities densities_i(z) and densities_j(zs)

import sympy as sp
import numpy as np
from pathlib import Path

def free_energy_EMF_z(ctx):
    """
    Enhanced Mean-Field (EMF) free-energy functional in two-point form:

        F = 1/2 ∑_{i,j} ∫ dz ∫ dzs  densities_i(z)  V_ij(|z - zs|)  densities_j(zs) / volume_factor_j(zs)

    This version replaces one-point density densities_i(z) with a pair of
    densities densities_i(z) and densities_j(zs) representing two spatial points.
    """

    from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials

    # -------------------------
    # Setup
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)
    scratch.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    potential_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)

    species = sorted(hc_data.keys())
    nelement = len(species)
    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag = [hc_data[s]["flag"] for s in species]

    # -------------------------
    # Define densities at two spatial points z and zs
    
    
    # Two symbolic coordinate labels
    z, zs = sp.symbols("z zs", real=True)

    # Pure symbolic density variables
    densities_z  = [sp.symbols(f"densities_z_{i}")  for i in range(nelement)]
    densities_zs = [sp.symbols(f"densities_zs_{i}") for i in range(nelement)]

    # ----------------------------
    # Pair interaction V_ij(|z - zs|)
    # ----------------------------
    vij = [[sp.symbols(f"v_{i}_{j}") for j in range(nelement)] for i in range(nelement)]

    # -------------------------
    # Volume correction factor at point zs
    # -------------------------
    volume_factor_zs = []
    for j in range(nelement):
        vf = 1
        for i in range(nelement):
            if flag[i] == 1 and j != i:
                avg_p_vol = (0.5*(sigmai[i] + sigmai[j]))**3
                term = (
                    sp.Rational(1,2) * densities_zs[i] * (sp.pi/6) * avg_p_vol
                    - sp.Rational(3,8) * (densities_zs[i] * (sp.pi/6) * avg_p_vol)**2
                )
                vf -= term
        volume_factor_zs.append(vf)

    # -------------------------
    # EMF free energy density (two‑point)
    # -------------------------
    f_mf = 0
    for i in range(nelement):
        for j in range(nelement):
            f_mf += sp.Rational(1,2) * densities_z[i] * vij[i][j] * densities_zs[j] / volume_factor_zs[j]

    # Do not integrate — leave in kernel form

    return {
        "species": species,
        "sigma_eff": sigmai,
        "flag": flag,
        "densities_z": densities_z,
        "densities_zs": densities_zs,
        "vij": vij,
        "volume_factor_zs": volume_factor_zs,
        "f_mf_two_point": f_mf,
    }

