# cdft_solver/external/external_potential_from_dict.py

"""
External Potential Generator (Dictionary-Driven)

Computes the potential on a spatial grid due to external species
defined in a nested dictionary. Uses isotropic pair potentials.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from cdft_solver.generators.potential.pair_potential_isotropic import pair_potential_isotropic as ppi


def external_potential_grid(
    ctx=None,
    data_dict=None,
    grid_coordinates=None,
    export_json=True,
    filename="supplied_data_external_potential.json",
    plot=False
):
    """
    Compute external potentials from a dictionary and a spatial grid.

    Parameters
    ----------
    ctx : object, optional
        Provides ctx.scratch_dir for exporting JSON.
    data_dict : dict
        Dictionary containing `system` -> `external` -> `species` and `interaction`.
    grid_points : int or list of int
        Number of points in each spatial direction (for info only if grid_coordinates provided).
    grid_coordinates : np.ndarray
        Nx3 array of [x, y, z] coordinates where potential should be evaluated.
    export_json : bool
        If True, export the result as JSON.
    filename : str
        Name of the JSON file if exported.

    Returns
    -------
    dict
        {
            "species": [...],
            "grid_coordinates": [[x, y, z], ...],
            "external_potentials": {
                "a": [...],  # potential values at each grid point
                "b": [...],
                ...
            }
        }
    """

    if data_dict is None:
        raise ValueError("data_dict must be provided")
    if grid_coordinates is None:
        raise ValueError("grid_coordinates must be provided as Nx3 array")

    # -------------------------
    # Recursive search helper
    # -------------------------
    def find_key_recursive(d, key):
        if not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        return None
        
    def find_system_species(d):
        """
        Recursively search for the first 'species' key
        that is NOT inside an 'external' dictionary.
        """
        if not isinstance(d, dict):
            return None

        for k, v in d.items():
            if k == "external":
                continue
            if k == "species" and isinstance(v, list):
                return v
            if isinstance(v, dict):
                found = find_system_species(v)
                if found is not None:
                    return found
        return None


    # -------------------------
    # Extract system & external info
    # -------------------------
    

    external = find_key_recursive(data_dict, "external")
    if not external:
        raise KeyError("No 'external' key found in system")

    external_species = external.get("species")
    if external_species is None:
        raise KeyError("No 'species' defined under 'external'")

    # System species (for interaction check)
    system_species = find_system_species(data_dict)
    if not system_species:
        raise KeyError("No system species found outside 'external'")

    # Check all interactions are defined
    interactions = external.get("interaction", {})
    for s in system_species:
        for es in external_species if isinstance(external_species, list) else [external_species]:
            pair_key = f"{s}{es}" if f"{s}{es}" in interactions else f"{es}{s}"
            if pair_key not in interactions:
                raise ValueError(f"Missing interaction definition for pair {s}-{es}")

    # -------------------------
    # Compute external potential
    # -------------------------
    grid_coordinates = np.array(grid_coordinates)
    N_grid = grid_coordinates.shape[0]

    external_potentials = {s: np.zeros(N_grid, dtype=float) for s in system_species}

    # For each external species, get positions and potentials
    for es in external_species if isinstance(external_species, list) else [external_species]:
        es_positions = np.array(external.get(es, {}).get("position", []))
        if es_positions.size == 0:
            raise ValueError(f"No positions defined for external species {es}")

        for s in system_species:
            pair_key = f"{s}{es}" if f"{s}{es}" in interactions else f"{es}{s}"
            pot_dict = interactions[pair_key]
            pot_func = ppi(pot_dict)

            # Compute distances from all external particles of this species
            for pos in es_positions:
                r_vec = grid_coordinates - pos  # Nx3
                r_mag = np.linalg.norm(r_vec, axis=1)
                external_potentials[s] += pot_func(r_mag)

    # -------------------------
    # Prepare output
    # -------------------------
    result = {
        "species": system_species,
        "grid_coordinates": grid_coordinates.tolist(),
        "external_potentials": {s: external_potentials[s].tolist() for s in system_species}
    }

    if export_json and ctx is not None:
        scratch_dir = Path(ctx.scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ External potentials exported to {out_file}")
        
        
    if plot and ctx is not None:
        scratch_dir = Path(ctx.scratch_dir)
        plot_dir = scratch_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for s in system_species:
            plt.figure(figsize=(6, 4))
            plt.scatter(range(N_grid), external_potentials[s], s=5, c='blue')
            plt.title(f"External Potential for Species '{s}'")
            plt.xlabel("Grid point index")
            plt.ylabel("Potential")
            plt.grid(True)
            
            y_min = -10
            y_max = 10
            plt.ylim(y_min, y_max)
            
            plot_file = plot_dir / f"external_potential_{s}.png"
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"✅ Plot saved: {plot_file}")



    return result

