import json
import numpy as np
from pathlib import Path
from cdft_solver.generators.potential.pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)


def meanfield_potentials(
    ctx=None,
    data_dict=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_mf.json",
    export_file=True
):
    """
    Mean-field potential generator using a dictionary input.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing species and interactions.
    ctx : object, optional
        Used only for scratch_dir when exporting JSON.
    mode : str
        'meanfield' or 'raw'
    grid_points : int
        Number of r-grid points.
    file_name_prefix : str
        Output JSON filename.
    export_file : bool
        If True, export JSON to scratch_dir.

    Returns
    -------
    dict
        {
            "species": [...],
            "interactions": converted_interactions,
            "potentials": {
                "aa": {
                    "r": [...],
                    "U_mf": [...]
                },
                ...
            }
        }
    """

    # ---------------------------------------------------------
    # Recursive discovery helpers
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

    species = find_key_recursive(data_dict, "species")
    interactions = find_key_recursive(data_dict, "interactions")

    if species is None or interactions is None:
        raise KeyError("Could not locate 'species' or 'interactions' in input dictionary.")

    # ---------------------------------------------------------
    # RAW MODE → return unchanged interactions
    # ---------------------------------------------------------
   
    # ---------------------------------------------------------
    # Mean-field conversion rules
    # ---------------------------------------------------------
    def convert_potential(potential):
        ptype = potential.get("type", "").lower()

        if ptype in ("hc", "ghc"):
            return {"type": "zero_potential"}

        elif ptype == "lj":
            new_pot = potential.copy()
            new_pot["type"] = "salj"
            return new_pot

        elif ptype == "mie":
            new_pot = potential.copy()
            new_pot["type"] = "ma"
            return new_pot

        return potential.copy()

    levels = ["primary", "secondary", "tertiary"]
    converted = {}

    for level in levels:
        if level not in interactions:
            continue
        converted[level] = {}
        for pair, pot in interactions[level].items():
            converted[level][pair] = convert_potential(pot)

    # ---------------------------------------------------------
    # Collect all unique pairs
    # ---------------------------------------------------------
    all_pairs = set()
    for lvl in converted.values():
        all_pairs.update(lvl.keys())

    # ---------------------------------------------------------
    # Compute TOTAL mean-field potentials
    # ---------------------------------------------------------
    potentials = {}

    for pair in sorted(all_pairs):

        cutoff = 0.0
        for lvl in converted:
            if pair in converted[lvl]:
                inter = converted[lvl][pair]
                cutoff = max(
                    cutoff,
                    inter.get("cutoff", inter.get("sigma", 1.0) * 5.0)
                )

        r = np.linspace(1e-5, cutoff, grid_points)
        u_total = np.zeros_like(r)

        for lvl in converted:
            if pair not in converted[lvl]:
                continue

            inter = converted[lvl][pair]
            if inter.get("type") == "zero_potential":
                continue

            V = ppi(inter)
            u_total += V(r)

        potentials[pair] = {
            "r": r.tolist(),
            "U_mf": u_total.tolist()
        }

    # ---------------------------------------------------------
    # Final output dictionary
    # ---------------------------------------------------------
    result = {
        "species": species,
        "mf_interactions": converted,
        "potentials": potentials
    }

    # ---------------------------------------------------------
    # Export JSON if requested
    # ---------------------------------------------------------
    if export_file and ctx is not None:
        out = Path(ctx.scratch_dir) / file_name_prefix
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Mean-field JSON exported: {out}")

    return result

