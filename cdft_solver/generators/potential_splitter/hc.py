import json
import numpy as np
from pathlib import Path
from scipy.integrate import simpson
from collections.abc import Mapping
from cdft_solver.generators.potential.pair_potential_isotropic import pair_potential_isotropic as ppi


def hard_core_potentials(
    ctx=None,
    input_data=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_hc.json",
    export_files=True
):
    """
    Hard-core detection module using a dictionary input instead of a JSON file.

    Returns a dictionary containing:
    - species, sigma, flag
    - potentials: {pair: {"r": [...], "U": [...]}}
    """
    beta=1.0
    # -------------------------
    # Recursive search helpers
    # -------------------------
   

    def barker_henderson_diameter(r, u):
        integrand = 1.0 - np.exp(-beta * np.clip(u, -100, 100))
        return simpson(integrand, r)

    def is_hard_core_type(ptype):
        return ptype.lower() in {"hc", "ghc", "hardcore", "lj", "mie"}

    # -------------------------
    # Locate species & interactions
    # -------------------------

    def find_key_recursive(obj, key):
        """
        Recursively find a key in nested mappings (dict, OrderedDict, etc).
        """
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found = find_key_recursive(item, key)
                if found is not None:
                    return found
        return None
  
  
  
    species = find_key_recursive(input_data, "species")
    interactions = find_key_recursive(input_data, "interactions")
    if species is None or interactions is None:
        raise KeyError("Could not find 'species' or 'interactions' in the provided dictionary.")

    levels = ["primary", "secondary", "tertiary"]
    n = len(species)
    idx = {sp: i for i, sp in enumerate(species)}

    sigma = np.zeros((n, n), dtype=float)
    flag = np.zeros((n, n), dtype=int)
    explicit = np.zeros((n, n), dtype=bool)

    # Dictionary to store potentials
    potentials_dict = {}

    # -------------------------
    # Detect explicit hard cores
    # -------------------------
    for level in levels:
        for key, inter in interactions.get(level, {}).items():
            sp_i, sp_j = key[0], key[1]
            i, j = idx[sp_i], idx[sp_j]

            ptype = inter["type"].lower()
            if not is_hard_core_type(ptype):
                continue

            # Direct hard-core
            if ptype in {"hc", "ghc", "hardcore"}:
                s = inter.get("sigma", 0.0)
                sigma[i, j] = sigma[j, i] = s
                
                if ptype in {"hc", "hardcore"}:
                    flag [i, j] = flag [j, i] = 1
                explicit[i, j] = explicit[j, i] = True
                continue

            # Soft-core → detect via potential
            pot = ppi(inter.copy())
            r = np.linspace(1e-5, inter.get("sigma", 1.0), grid_points)
            u = np.clip(pot(r), -1e3, 1e7)

            if np.any(u[:5] > 1e6):
                s = barker_henderson_diameter(r, u)
                if (i==j):
                    flag [i, j] = flag [j, i] = 1
                sigma[i, j] = sigma[j, i] = s
                explicit[i, j] = explicit[j, i] = True


    # -------------------------
    # Off-diagonal additive rule
    # -------------------------
    
    if np.any(explicit):
        for i in range(n):
            j=i
            if explicit[i, j]:
                continue

            found = False
            for k in range(n):
                if explicit[i, k] and explicit[k, k]:
                    sigma[i, i] = 2 * sigma[i, k] - sigma[k, k]  # reverse additive rule
                    explicit[i, i] = True
                    found = True
                    break
            if not found:
                raise ValueError(f"Cannot determine σ({species[i]},{species[j]})")
            
                        
        for i in range(n):
            for j in range(i+1, n):
                # Off-diagonal additive rule
                if explicit[i, i] and explicit[j, j]:
                    sigma[i, j] = sigma[j, i] = 0.5 * (sigma[i, i] + sigma[j, j])
                    explicit[i, j] = explicit[j, i] = True
                else:
                    raise ValueError(f"Cannot determine σ({species[i]},{species[j]})")


    
    
    
    # -------------------------
    # Build potentials dictionary
    # -------------------------
    for i in range(n):
        for j in range(i, n):
            sp_i, sp_j = species[i], species[j]
            s = sigma[i, j]
            cutoff = s * 5.0
            r_values = np.linspace(1e-5, cutoff, grid_points)
            u_values = np.where(r_values < s, 1e12, 0.0)
            potentials_dict[f"{sp_i}{sp_j}"] = {
                "r": r_values.tolist(),
                "U": u_values.tolist()
            }

    # -------------------------
    # Prepare output dictionary
    # -------------------------
    hc_data = {
        "species": species,
        "sigma": sigma.tolist(),
        "flag": flag.tolist(),
        "potentials": potentials_dict
    }

    # Export JSON if requested
    if export_files and ctx is not None:
        out_json = Path(ctx.scratch_dir) / file_name_prefix
        with open(out_json, "w") as f:
            json.dump(hc_data, f, indent=2)
        print(f"✅ Exported HC to JSON: {out_json}")

    return hc_data
