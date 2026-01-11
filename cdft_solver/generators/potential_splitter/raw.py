import json
import numpy as np
from pathlib import Path
import os
from scipy.interpolate import interp1d

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

    if not species:
        raise KeyError("Could not locate 'species' in input dictionary.")

    N = len(species)
    r_min = 0.0
    default_grid_max = 10.0

    result = {"species": species, "potentials": {}}

    # =========================================================
    # CASE 1: interactions supplied
    # =========================================================
    if interactions is not None:
        levels = ["primary", "secondary", "tertiary"]

        # Collect interactions per pair
        pair_dict = {}
        for level in levels:
            for pair, inter in interactions.get(level, {}).items():
                pair_dict.setdefault(pair, []).append(inter)

        for pair, inter_list in pair_dict.items():

            # -------------------------------
            # Determine grid_max
            # -------------------------------
            grid_max = default_grid_max

            for inter in inter_list:
                # Analytic
                if "type" in inter:
                    if "cutoff" in inter:
                        grid_max = max(grid_max, inter["cutoff"])
                    elif "sigma" in inter:
                        grid_max = max(grid_max, 5.0 * inter["sigma"])

                # File-based
                elif "filename" in inter:
                    filepath = os.path.join(os.getcwd(), inter["filename"])
                    if not os.path.isfile(filepath):
                        raise FileNotFoundError(f"Potential file not found: {filepath}")

                    data = np.loadtxt(filepath)
                    if data.shape[1] < 2:
                        raise ValueError("Tabulated potential must have at least two columns (r, U)")

                    grid_max = max(grid_max, data[:, 0].max())

            # -------------------------------
            # Build grid
            # -------------------------------
            r = np.linspace(r_min, grid_max, grid_points)
            u_total = np.zeros_like(r)

            # -------------------------------
            # Accumulate raw potentials
            # -------------------------------
            for inter in inter_list:

                # Analytic
                if "type" in inter:
                    V = ppi(inter)
                    u_total += V(r)

                # Tabulated
                elif "filename" in inter:
                    data = np.loadtxt(inter["filename"])
                    r_tab = data[:, 0]
                    u_tab = data[:, 1]

                    interp_u = interp1d(
                        r_tab,
                        u_tab,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(u_tab[0], 0.0)
                    )
                    u_total += interp_u(r)

            result["potentials"][pair] = {
                "r": r.tolist(),
                "U": u_total.tolist()
            }

    # =========================================================
    # CASE 2: no interactions → zero potentials
    # =========================================================
    else:
        r = np.linspace(r_min, default_grid_max, grid_points)
        u_zero = np.zeros_like(r)

        for i in range(N):
            for j in range(i, N):
                pair = f"{species[i]}-{species[j]}"
                result["potentials"][pair] = {
                    "r": r.tolist(),
                    "U": u_zero.tolist()
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

