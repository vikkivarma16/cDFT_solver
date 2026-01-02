import json
import numpy as np
from pathlib import Path
from cdft_solver.generators.potential.pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)
from cdft_solver.generators.potential_splitter.mf_potential_registry import (
    convert_potential_via_registry,
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
        raise KeyError(
            "Could not locate 'species' or 'interactions' in input dictionary."
        )

    # ---------------------------------------------------------
    # Mean-field conversion via REGISTRY
    # ---------------------------------------------------------
    levels = ["primary", "secondary", "tertiary"]
    converted = {}

    for level in levels:
        if level not in interactions:
            continue

        converted[level] = {}
        for pair, pot in interactions[level].items():
            converted[level][pair] = convert_potential_via_registry(pot)

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

