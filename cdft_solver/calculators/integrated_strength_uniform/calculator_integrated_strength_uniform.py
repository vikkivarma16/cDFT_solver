import numpy as np
from collections import defaultdict
from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials
from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi
from cdft_solver.generators.potential_splitter.generator_potential_total import raw_potentials

# -------------------------
# Bulk potential builder (centralized using ppi)
# -------------------------
def build_bulk_potential_matrix(ctx, r, r_max):
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    species = pot_data["species"]
    interactions_by_level = pot_data.get("interactions", {}) or raw_potentials
    levels = ["primary", "secondary", "tertiary"]

    N = len(species)
    Nr = len(r)
    u_matrix = np.zeros((N, N, Nr))

    for level in levels:
        level_dict = interactions_by_level.get(level, {}) or {}
        for pair_key, params in level_dict.items():
            if len(pair_key) != 2:
                continue
            a, b = pair_key[0], pair_key[1]
            try:
                i = species.index(a)
                j = species.index(b)
            except ValueError:
                continue

            potential_dict = params.copy() if isinstance(params, dict) else dict(params)
            cutoff = potential_dict.get("cutoff", r_max)

            # Centralized potential function
            V_func = ppi(potential_dict)
            V_r_local = V_func(r)

            u_matrix[i, j, :] += V_r_local
            if i != j:
                u_matrix[j, i, :] += V_r_local

    return species, u_matrix

# -------------------------
# Compute vk using g_r = 1
# -------------------------
def compute_vk_meanfield(r, u_matrix, species):
    N = len(species)
    vk = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            integrand = u_matrix[i, j, :] * 4.0 * np.pi * r**2
            vk[i, j] = np.trapz(integrand, r)
    return vk

# -------------------------
# Top-level uniform vk function
# -------------------------
def vk_uniform(ctx):
    """
    Load all meanfield potentials from context and compute vk assuming g_r = 1.
    Bulk potential matrix built from centralized pair_potential_isotropic generator.
    """
    r = np.linspace(1e-3, 5.0, 400)
    species, u_matrix = build_bulk_potential_matrix(ctx, r, r_max=5.0)
    vk = compute_vk_meanfield(r, u_matrix, species)
    return {"species": species, "vk": vk}

