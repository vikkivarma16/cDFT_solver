import json
import numpy as np
from pathlib import Path
from cdft_solver.generators.potential.pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)


def raw_potentials(
    ctx=None,
    input_data=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_raw.json",
    export_files=True
):    
    """
    Process raw interaction potentials from a dictionary
    and return JSON-serializable data only.

    Parameters
    ----------
    input_data : dict
        Dictionary containing particle interactions.
    ctx : object, optional
        Used only for scratch_dir when exporting.
    grid_points : int
        Number of r-grid points.
    file_name_prefix : str
        Output JSON filename.
    export_files : bool
        If True, export JSON to scratch_dir.

    Returns
    -------
    dict
        {
            "potentials": {
                "aa": {
                    "r": [...],
                    "U_raw": [...]
                },
                ...
            }
        }
    """

    # ---------------------------------------------------------
    # Recursive interaction discovery
    # ---------------------------------------------------------
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

    species = find_key_recursive(input_data, "species")
    interactions = find_key_recursive(input_data, "interactions")

    if not interactions:
        raise KeyError("Could not locate 'interactions' in input dictionary.")

    levels = ["primary", "secondary", "tertiary"]

    # ---------------------------------------------------------
    # Collect interactions per pair
    # ---------------------------------------------------------
    pair_dict = {}
    for level in levels:
        for pair, inter in interactions.get(level, {}).items():
            pair_dict.setdefault(pair, []).append(inter)

    # ---------------------------------------------------------
    # Compute raw potentials
    # ---------------------------------------------------------
    result = {"species": species, "potentials": {}}

    for pair, inter_list in pair_dict.items():

        cutoff = max(
            inter.get("cutoff", inter.get("sigma", 1.0) * 5.0)
            for inter in inter_list
        )

        r = np.linspace(1e-5, cutoff, grid_points)
        u_total = np.zeros_like(r)

        for inter in inter_list:
            V = ppi(inter)
            u_total += V(r)

        # Store as JSON-safe lists
        result["potentials"][pair] = {
            "r": r.tolist(),
            "U": u_total.tolist()
        }

    # ---------------------------------------------------------
    # Export JSON if requested
    # ---------------------------------------------------------
    if export_files and ctx is not None:
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out = scratch / file_name_prefix
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Exported raw potential to JSON: {out}")

    return result

