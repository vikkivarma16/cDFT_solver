# cdft_solver/isochores/fmt_weights.py

"""
FMT Weight Function Generator (Dictionary-Driven)

Generates 1D FMT weight functions in k-space based on particle sizes.
Accepts a dictionary with particle interaction parameters instead of JSON.
Optionally plots the weights.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

EPSILON = 1e-10
PI = np.pi

def fmt_weights_planer(
    ctx=None,
    data_dict=None,
    grid_properties=None,
    export_json=True,
    filename="supplied_data_weight_FMT_k_space.json",
    plot=False
):
    """
    Generate 1D FMT weight functions for cylindrical geometry.

    Parameters
    ----------
    ctx : object
        Must provide ctx.scratch_dir (output folder) and optionally ctx.plots_dir
    data_dict : dict
        Dictionary containing particle interactions with `species` and `sigma`
    k_space_coordinates : np.ndarray
        Nx3 array of k-space points
    export_json : bool
        Whether to export the weight functions as JSON
    filename : str
        Output JSON filename
    plot : bool
        Whether to generate plots

    Returns
    -------
    dict
        {
            "species": [...],
            "k_space": [[kx, ky, kz], ...],
            "weight_functions": {
                "a": [...],
                "b": [...],
                ...
            }
        }
    """

    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx with scratch_dir must be provided")
    if data_dict is None:
        raise ValueError("data_dict must be provided")
    

    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if plot and hasattr(ctx, "plots_dir"):
        plot_dir = Path(ctx.plots_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Recursive search helper
    # -------------------------
    # -------------------------
    # Extract species and sigma
    # -------------------------
    species = data_dict["species"]
    sigma_matrix = np.asarray(data_dict["sigma"])

    n_species = len(species)

    # Validate sigma matrix
    if sigma_matrix.shape != (n_species, n_species):
        raise ValueError(
            f"sigma must be a {n_species}×{n_species} matrix, "
            f"got shape {sigma_matrix.shape}"
        )

    # Particle sizes = diagonal elements
    particle_sizes = {
        species[i]: float(sigma_matrix[i, i])
        for i in range(n_species)
    }

    # -------------------------
    # Prepare k-space
    # -------------------------
    
    k_space_coordinates = np.array(grid_properties["k_space"])  # shape (N, 3)
    
    N_k = k_space_coordinates.shape[0]
    dimension = 1  # cylindrical 1D weight function

    # -------------------------
    # Calculate weight functions
    # -------------------------
    weight_functions = {}
    for particle, size in particle_sizes.items():
        weight_list = []
        for kx, ky, kz in k_space_coordinates:
            k_value = kx
            weight_vector = [kx, ky, kz]
            if np.abs(k_value) < EPSILON:
                # Avoid division by zero
                weight_vector.extend([
                    1.0,
                    size * 0.5,
                    PI * size**2,
                    PI * size**3 / 6.0,
                    0.0,
                    0.0
                ])
            else:
                weight_vector.extend([
                    np.sin(k_value * PI * size) / (k_value * size * PI),
                    np.sin(k_value * PI * size) / (2.0 * k_value * PI),
                    size * np.sin(k_value * PI * size) / k_value,
                    (np.sin(k_value * PI * size) / (2.0 * k_value**3 * PI**2)
                     - size * np.cos(k_value * PI * size) / (2.0 * k_value**2 * PI)),
                    1j*(k_value * PI * size * np.cos(k_value * PI * size) - np.sin(k_value * PI * size)) / (2.0 * size * PI**2 * k_value**2),
                    1j*(k_value * PI * size * np.cos(k_value * PI * size) - np.sin(k_value * PI * size)) / (k_value**2 * PI)
                ])
            weight_list.append(np.array(weight_vector))
        weight_functions[particle] = np.array(weight_list)

    # -------------------------
    # Export JSON
    # -------------------------
    result = {
        "species": list(particle_sizes.keys()),
        "k_space": k_space_coordinates.tolist(),
        "weight_functions": {p: weight_functions[p].tolist() for p in particle_sizes}
    }

    if export_json:
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ FMT weight functions exported to {out_file}")

    # -------------------------
    # Optional plotting
    # -------------------------
    if plot and hasattr(ctx, "plots_dir"):
        k_vals = k_space_coordinates[:, 0]
        for particle, weight_array in weight_functions.items():
            Vk_abs = np.abs(weight_array[:, 3])  # example: 4th component
            plt.figure(figsize=(7,5))
            plt.plot(k_vals, Vk_abs, label=f"|V(k)| {particle}")
            plt.xlabel("k")
            plt.ylabel("|V(k)|")
            plt.title(f"FMT Weight Function for {particle}")
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_dir / f"FMT_weight_{particle}.png", dpi=300)
            plt.close()
            print(f"✅ Plot saved: {plot_dir / f'FMT_weight_{particle}.png'}")

    return result

