import json
import numpy as np
from pathlib import Path
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Import your existing potential generator
from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi


def hard_core_potentials(ctx, save_arrays=False, output_dir="hard_core_data", beta=1.0, Nr=4000):
    """
    Hard-core potential splitter module.
    Computes effective hard-core diameters (σ_eff) using Barker–Henderson integration
    over the isotropic pair potential generated via pair_potential_isotropic.

    If no hard-core or excluded-volume types are found, returns sigma=0.0 and flag=0 for all species.
    """

    def barker_henderson_diameter_from_u(r_values, u_values, beta=1.0):
        """Compute Barker–Henderson effective hard-core diameter."""
        integrand = 1.0 - np.exp(-beta * np.clip(u_values, a_min=-100, a_max=100))
        return simpson(integrand, r_values)

    def is_hard_core_type(ptype):
        """Return True if potential type implies excluded volume or hard core."""
        return ptype.lower() in ["hc", "ghc", "lj", "mie"]

    def load_interaction_data(ctx):
        """Load and parse the particle interaction data from scratch directory or fallback file."""
        scratch = Path(ctx.scratch_dir)
        particle_json_in_scratch = scratch / "input_data_particles_interactions_parameters.json"
        particle_json_path = particle_json_in_scratch if particle_json_in_scratch.exists() else getattr(ctx, "input_file", None)

        if particle_json_path is None or not Path(particle_json_path).exists():
            raise FileNotFoundError(f"Particle interactions JSON not found in {scratch} or fallback path.")

        with open(particle_json_path, "r") as f:
            data = json.load(f)

        return data["particles_interactions_parameters"]

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    params = load_interaction_data(ctx)
    species = params["species"]
    interactions = params["interactions"]

    hard_core_flags = {sp: False for sp in species}
    hard_core_sizes = {sp: None for sp in species}
    potential_arrays = {}

    # -------------------------------------------------------------------------
    # Compute σ_eff for each species using same potential logic as vk_rdf
    # -------------------------------------------------------------------------
    for sp in species:
        for level in ["primary", "secondary", "tertiary"]:
            key = f"{sp}{sp}"
            if key in interactions.get(level, {}):
                inter = interactions[level][key]
                ptype = inter["type"].lower()
                sigma = inter.get("sigma", 1.0)
                cutoff = inter.get("cutoff", 5.0)

                # Case 1: Direct hard-core types
                if ptype in ["hc", "ghc", "hardcore"]:
                    d_eff = sigma
                    hard_core_flags[sp] = True if ptype == "hc" else False
                    hard_core_sizes[sp] = d_eff
                    continue
                    

                # Case 2: Soft-core → generate potential via ppi (same as vk_rdf)
                # Case 2: Soft-core or hard-core potential detection
                potential_dict = inter.copy()
                V_func = ppi(potential_dict)

                r_values = np.linspace(1e-5, sigma, Nr)
                u_values = V_func(r_values)

                # Safety clip: prevent overflow in exp(-βu)
                u_values = np.clip(u_values, a_min=-1e3, a_max=1e7)

                # Detect hard-core behaviour:
                # If potential near r_min exceeds 1e6, treat as hard-core
                if np.any(u_values[:5] > 1e6):  # check first few points near r → 0
                    # Compute Barker–Henderson effective diameter
                    d_eff = barker_henderson_diameter_from_u(r_values, u_values, beta)
                    hard_core_sizes[sp] = d_eff
                    hard_core_flags[sp] = True
                else:
                    # No strong repulsion — treat as soft or ideal interaction
                    hard_core_sizes[sp] = 0.0
                    hard_core_flags[sp] = False


                if save_arrays:
                    potential_arrays[sp] = {"r": r_values.tolist(), "u": u_values.tolist()}

    # -------------------------------------------------------------------------
    # Cross-consistency check
    # -------------------------------------------------------------------------
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            key1, key2 = f"{sp_i}{sp_j}", f"{sp_j}{sp_i}"
            inter = None
            for level in ["primary", "secondary", "tertiary"]:
                if key1 in interactions[level]:
                    inter = interactions[level][key1]
                    break
                elif key2 in interactions[level]:
                    inter = interactions[level][key2]
                    break
            if inter is None:
                continue

            if is_hard_core_type(inter["type"]):
                if hard_core_sizes[sp_i] is None or hard_core_sizes[sp_j] is None:
                    raise ValueError(
                        f"Inconsistent hard-core: pair {sp_i}-{sp_j} has excluded-volume potential "
                        f"but one species lacks a defined size."
                    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------



    print("\n✅ Hard-core consistency check passed.")    
    if all(flag is False for flag in hard_core_flags.values()):
        print("No hard-core or excluded volume detected — setting σ_eff = 0.0 and flag = 0 for all species.")
        summary = {sp: {"sigma_eff": 0.0, "flag": 0} for sp in species}
    else:
        summary = {
            sp: {"sigma_eff": hard_core_sizes[sp] if hard_core_sizes[sp] is not None else 0.0,
                 "flag": int(hard_core_flags[sp])}
            for sp in species
        }   
    for sp, val in summary.items():
        print(f"  {sp}: σ_eff = {val['sigma_eff']:.4f}, flag = {val['flag']}")



    # -------------------------------------------------------------------------
    # Optional: Save computed potential arrays
    # -------------------------------------------------------------------------
    if save_arrays and potential_arrays:
        output_path = Path(ctx.scratch_dir) / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        for sp, arr in potential_arrays.items():
            np.savez(output_path / f"{sp}_potential_arrays.npz", **arr)

    return summary

