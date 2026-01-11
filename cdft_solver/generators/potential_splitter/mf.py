import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import os

from cdft_solver.generators.potential.pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)
from cdft_solver.generators.potential_splitter.mf_registry import (
    convert_potential_via_registry,
)


def meanfield_potentials(
    ctx=None,
    input_data=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_mf.json",
    export_files=True
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

    species = find_key_recursive(input_data, "species")
    n =  len(species)
    interactions = find_key_recursive(input_data, "interactions")

    if species is None :
        raise KeyError(
            "Could not locate 'species' in input dictionary."
        )
    
    r_min = 0
    grid_max = 10.0  # default fallback if nothing else
    
    if interactions is not None :
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

        
        
        def wca_split(r, U):
            """Split a potential into repulsive (soft) and attractive parts (WCA)."""
            idx_min = np.argmin(U)
            r_min = r[idx_min]
            U_min = U[idx_min]

            U_rep = np.zeros_like(U)
            U_att = np.zeros_like(U)

            for i, ri in enumerate(r):
                if ri <= r_min:
                    U_rep[i] = U[i] - U_min
                    U_att[i] = U_min
                else:
                    U_rep[i] = 0.0
                    U_att[i] = U[i]

            return U_rep, U_att

        # ------------------------------
        # Compute mean-field potentials
        # ------------------------------
        potentials = {}

        for pair in sorted(all_pairs):

            # Determine r-grid for the pair
            

            # Loop over levels to determine maximum range
            for lvl in converted:
                if pair not in converted[lvl]:
                    continue
                inter = converted[lvl][pair]

                # Analytic potentials use sigma or cutoff if available
                if "type" in inter:
                    sigma = inter.get("sigma", None)
                    cutoff_val = inter.get("cutoff", None)

                    if cutoff_val is not None:
                        grid_max = max(grid_max, cutoff_val)
                    elif sigma is not None:
                        grid_max = max(grid_max, sigma * 5.0)
                    else:
                        grid_max = max(grid_max, 10.0)  # fallback

                # File-based potentials determine grid from the file itself
                elif "filename" in inter:
                    filepath = os.path.join(os.getcwd(), inter["filename"])
                    if not os.path.isfile(filepath):
                        raise FileNotFoundError(f"Potential file not found: {filepath}")

                    data = np.loadtxt(filepath)
                    if data.shape[1] < 2:
                        raise ValueError("Tabulated potential must have at least two columns: r U")

                    r_tab = data[:, 0]
                    u_tab = data[:, 1]

                    # Set grid_max to the **full extent of the file**
                    r_file_max = r_tab.max()

                    # Update overall grid_max
                    grid_max = max(grid_max, r_file_max)


            # Build uniform grid for integration
            r = np.linspace(r_min, grid_max, grid_points)
            u_total = np.zeros_like(r)

            for lvl in converted:
                if pair not in converted[lvl]:
                    continue

                inter = converted[lvl][pair]

                # Skip zero potential
                if inter.get("type") == "zero_potential":
                    continue

                # -------------------------------
                # Analytic potential
                # -------------------------------
                if "type" in inter:
                    V = ppi(inter)
                    u_raw = V(r)

                # -------------------------------
                # Tabulated potential
                # -------------------------------
                elif "filename" in inter:
                    filepath = os.path.join(os.getcwd(), inter["filename"])
                    if not os.path.isfile(filepath):
                        raise FileNotFoundError(f"Potential file not found: {filepath}")

                    data = np.loadtxt(filepath)
                    if data.shape[1] < 2:
                        raise ValueError("Tabulated potential must have at least two columns: r U")

                    r_tab = data[:, 0]
                    u_tab = data[:, 1]

                    V_interp = interp1d(
                        r_tab,
                        u_tab,
                        kind="linear",
                        bounds_error=False,
                        fill_value=(u_tab[0], 0.0)
                    )
                    u_raw = V_interp(r)

                else:
                    continue

                # -------------------------------
                # Check for hard-core
                # -------------------------------
                # We consider hard-core if potential diverges near r ~ 0
                short_range = u_raw[r < (r.min() + 0.05 * (r.max() - r.min()))]
                has_hc = np.any(short_range > 1e5)

                # -------------------------------
                # WCA splitting only if hard-core exists
                # -------------------------------
                if has_hc:
                    u_soft, _ = wca_split(r, u_raw)
                    u_total += u_soft
                else:
                    u_total += u_raw

            # Store final mean-field potential
            potentials[pair] = {
                "r": r.tolist(),
                "U": u_total.tolist()
            }
    else:
        r = np.linspace(r_min, grid_max, grid_points)
        u_total = np.zeros_like(r)
        potentials = {}
        converted = {}
        converted["primary"] = {}
        for i in range(n):
            for j in range(i, n):
                pair = f"{species[i]}{species[j]}"
                potentials[pair] = {
                        "r": r.tolist(),
                        "U": u_total.tolist()
                    }
                converted["primary"][pair] =  {"type": "zero_potential", "sigma" : 0.0, "cutoff": 5, "epsilon" : 0.0}
                
                
            


    
    

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
    if export_files and ctx is not None:
        out = Path(ctx.scratch_dir) / file_name_prefix
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Exported Mean-field to JSON: {out}")

    return result

