import numpy as np
import sympy as sp
import json
from pathlib import Path
from scipy.interpolate import interp1d
from collections.abc import Mapping
from collections.abc import Mapping
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials

def vij_radial_kernel(
    ctx,
    config,
    kernel,
    supplied_data = None,
    export_json=False,
    filename="vij_integrated.json",
):
    """
    Compute v_ij = ∫ 4π r^2 K_ij(r) U_ij(r) dr
    with grid mismatch handled by interpolation.

    Parameters
    ----------
    ctx : object
        Must have ctx.scratch_dir if export_json=True
    config : dict
        Input configuration for mean-field potentials
    kernel : dict
        {(s1, s2): {"r": array, "values": array}}
    r_min, r_max : float, optional
        Integration bounds
    n_grid : int
        Number of points in common grid
    export_json : bool
        Export numeric results to JSON
    filename : str
        JSON output file

    Returns
    -------
    dict
        {
            "species": [...],
            "vij_symbols": [[sympy.Symbol]],
            "vij_numeric": {(s1, s2): float}
        }
    """

    # -------------------------
    # Recursive key search
    # -------------------------
    def find_key_recursive(obj, key):
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

    species = find_key_recursive(config, "species")
    species_names = species
    beta = find_key_recursive(config, "beta")
    if species is None:
        raise ValueError("Species list not found in config")

    n_species = len(species)
    n_grid=5000

    # -------------------------
    # Symbolic v_ij matrix
    # -------------------------
    vij_symbols = [
        [sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
        for i in range(n_species)
    ]

    vij_numeric = {}

    # -------------------------
    # Mean-field potentials (user must provide function or module)
    # -------------------------
    
    mf_data = meanfield_potentials(
        ctx=ctx,
        input_data=config,
        grid_points = n_grid,
        file_name_prefix="supplied_data_potential_mf.json",
        export_files=False
    )

    potential_dict = mf_data["potentials"]
    
    r = np.linspace(0, 5, n_grid)
    n = len(species)
    u_matrix = np.zeros((n, n, n_grid))

    for i, si in enumerate(species):
        for j in range(i, n):   # <-- only j >= i
            sj = species[j]

            key_ij = si + sj
            key_ji = sj + si

            pdata = (
                potential_dict.get(key_ij)
                or potential_dict.get(key_ji)
            )

            if pdata is None:
                raise KeyError(
                    f"Missing potential for pair '{si}-{sj}' "
                    f"(expected '{key_ij}' or '{key_ji}')"
                )

            # interpolate once
            interp_u = interp1d(
                pdata["r"],
                pdata["U"],
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )
            r = pdata["r"] 

            u_val = beta * interp_u(r)

            # symmetric assignment
            u_matrix[i, j, :] = u_val
            u_matrix[j, i, :] = u_val
            
            
           
    
    
    
    U_dict = {}
    for i, si in enumerate(species_names):
        for j, sj in enumerate(species_names):
            U_dict[(si, sj)] = {"r": r, "U": u_matrix[i, j]}
    
    

    # -------------------------
    # Loop over species pairs (use symmetry)
    # -------------------------
    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):  # j >= i
            # Try both orders
            key = (si, sj)
            rkey = (sj, si)

            if key in kernel and key in U_dict:
                ker_data = kernel[key]
                U_data   = U_dict[key]
            elif rkey in kernel and rkey in U_dict:
                ker_data = kernel[rkey]
                U_data   = U_dict[rkey]
            else:
                raise KeyError(f"Missing kernel or U for pair {si}-{sj}")

            rk = np.asarray(ker_data["r"], dtype=float)
            K  = np.asarray(ker_data["values"], dtype=float)

            ru = np.asarray(U_data["r"], dtype=float)
            Uv = np.asarray(U_data["U"], dtype=float)

            # -------------------------
            # Common grid
            # -------------------------
            r_lo = max(rk.min(), ru.min())
            r_hi = min(rk.max(), ru.max())
            if r_hi <= r_lo:
                raise ValueError(f"No overlapping r-domain for pair {si}-{sj}")

            r_common = np.linspace(r_lo, r_hi, n_grid)

            # -------------------------
            # Interpolation
            # -------------------------
            Kc = interp1d(rk, K, kind="linear", bounds_error=False, fill_value=0.0)(r_common)
            Uc = interp1d(ru, Uv, kind="linear", bounds_error=False, fill_value=0.0)(r_common)
            # -------------------------
            # Radial integral
            # -------------------------
            vij_val = float(np.trapz(4.0 * np.pi * r_common**2 * Kc * Uc, r_common))
            
            print( vij_val )

            # Assign symmetric
            vij_numeric[key] = vij_val
            vij_numeric[rkey] = vij_val

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json:
        if ctx is None or not hasattr(ctx, "scratch_dir"):
            raise ValueError("ctx with scratch_dir required for JSON export")

        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out_file = scratch / filename

        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "vij": {f"{k[0]}_{k[1]}": v for k, v in vij_numeric.items()}
                },
                f,
                indent=4,
            )
        print(f"✅ Integrated v_ij exported to {out_file}")

    return {
        "species": species,
        "vij_symbols": vij_symbols,
        "vij_numeric": vij_numeric,
    }

