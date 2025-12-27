# Corrected symbolic two-point style version of free_energy_void_z
# Fully consistent with the SMF two-point symbolic style (densities_z_i, densities_zs_i)

import sympy as sp
import numpy as np
from pathlib import Path

def free_energy_void_z(ctx):
    """
    Cavity Mean-Field (CMF) free-energy kernel in symbolic two-point form.

    Unlike the EMF version, here we apply void-volume corrections but keep the
    kernel strictly symbolic using densities_z_i and densities_zs_i.
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
    # Load potentials and HC data
    # -------------------------
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)

    species = sorted(hc_data.keys())
    nelement = len(species)

    sigmai = [hc_data[s]["sigma_eff"] for s in species]
    flag   = [hc_data[s]["flag"] for s in species]

    # -------------------------
    # Two symbolic spatial coordinates
    # -------------------------
    z, zs = sp.symbols("z zs", real=True)

    # -------------------------
    # Define two-point symbolic densities
    # -------------------------
    densities_z  = [sp.symbols(f"densities_z_{i}")  for i in range(nelement)]
    densities_zs = [sp.symbols(f"densities_zs_{i}") for i in range(nelement)]

    # -------------------------
    # Pair interaction kernels (symbolic)
    # -------------------------
    vij = [[sp.symbols(f"v_{i}_{j}") for j in range(nelement)] for i in range(nelement)]

    # -------------------------
    # Compute void-volume correction factors at z and zs
    # -------------------------
    # Here we only apply the correction at *one* point (zs), consistent with
    # a cavity correction acting on the receiving density.

    volume_factor_zs = []
    for j in range(nelement):
        vf = 1
        for i in range(nelement):
            if flag[i] == 1 and j != i:
                avg_p_vol = sigmai[i]**3
                term = (sp.pi/6) * avg_p_vol * densities_zs[i]
                vf -= term
        volume_factor_zs.append(vf)
        
        
    volume_factor_z = []
    for j in range(nelement):
        vf = 1
        for i in range(nelement):
            if flag[i] == 1 and j != i:
                avg_p_vol = sigmai[i]**3
                term = (sp.pi/6) * avg_p_vol * densities_z[i]
                vf -= term
        volume_factor_z.append(vf)

    # -------------------------
    # CMF free-energy kernel
    # -------------------------
    f_mf = 0
    for i in range(nelement):
        for j in range(nelement):
            f_mf += sp.Rational(1,2) * densities_z[i] * vij[i][j] * densities_zs[j] / (volume_factor_zs[j]*volume_factor_z[i])

    # -------------------------
    # Package symbolic result
    # -------------------------
    return {
        "species": species,
        "sigma_eff": sigmai,
        "flag": flag,
        "densities_z": densities_z,
        "densities_zs": densities_zs,
        "vij": vij,
        "volume_factor_zs": volume_factor_zs,
        "voluem_factor_z": volume_factor_z,
        "f_mf_two_point": f_mf,
    }
