# coexistence_isochem_fixed.py

import json
import ast
import numpy as np
import sympy as sp
from scipy.optimize import root
import random
from copy import deepcopy
from pathlib import Path
from cdft_solver.calculators.coexistence_densities.kernel_generator import build_strength_kernel
from cdft_solver.calculators.integrated_strength.integrated_strength_radial_kernal import vij_radial_kernel

# ============================================================
# GLOBAL
# ============================================================
CURRENT_VIJ_PER_PHASE = None


# ============================================================
# RECURSIVE SEARCH UTILITIES
# ============================================================
def deep_get(data, key, default=None):
    """
    Recursively search for key in nested dicts/lists.
    Returns first match.
    """
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            out = deep_get(v, key, default)
            if out is not default:
                return out
    elif isinstance(data, list):
        for item in data:
            out = deep_get(item, key, default)
            if out is not default:
                return out
    return default


def deep_get_all(data, key):
    """Return all matches for key in nested structure."""
    found = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                found.append(v)
            found.extend(deep_get_all(v, key))
    elif isinstance(data, list):
        for item in data:
            found.extend(deep_get_all(item, key))
    return found


def build_thermodynamics_from_fe_res(fe_res):
    """
    Given a free-energy result dictionary (from JSON),
    reconstruct SymPy objects and compute chemical potentials
    and pressure.

    Parameters
    ----------
    fe_res : dict
        Free energy JSON dictionary

    Returns
    -------
    dict with:
        - density_symbols
        - vij_symbols
        - free_energy
        - mu_expressions
        - pressure_expression
        - eval_mu_pressure(rho, vij)
    """

    # --------------------------------------------------------
    # 1) Reconstruct symbols
    # --------------------------------------------------------
    
    variables = deep_get(fe_res, "variables", [])
    expr_str = deep_get(fe_res, "expression")
    
    
    print (variables)

    if expr_str is None:
        raise ValueError("Free energy expression not found in fe_res")

    symbols = {}

    for v in variables:
        if isinstance(v, sp.Symbol):
            # Already a Symbol
            symbols[v.name] = v
        else:
            # Assume string-like
            symbols[str(v)] = sp.Symbol(str(v))


    # --------------------------------------------------------
    # 2) Reconstruct free-energy expression
    # --------------------------------------------------------
    f_sym = sp.sympify(expr_str, locals=symbols)

    # --------------------------------------------------------
    # 3) Identify density and interaction variables
    # --------------------------------------------------------
    density_syms = sorted(
        [s for s in symbols.values() if s.name.startswith("rho_")],
        key=lambda s: s.name
    )

    vij_syms = sorted(
        [s for s in symbols.values() if s.name.startswith("v_")],
        key=lambda s: s.name
    )

    if not density_syms:
        raise ValueError("No density variables (rho_*) found")

    # --------------------------------------------------------
    # 4) Chemical potentials
    # --------------------------------------------------------
    mu_syms = [sp.diff(f_sym, rho) for rho in density_syms]

    # --------------------------------------------------------
    # 5) Pressure
    #     P = -f + sum_i rho_i * mu_i
    # --------------------------------------------------------
    pressure_sym = -f_sym + sum(r * mu for r, mu in zip(density_syms, mu_syms))

    # --------------------------------------------------------
    # 6) Numerical evaluators
    # --------------------------------------------------------
    all_args = density_syms + vij_syms

    mu_funcs = [sp.lambdify(all_args, mu, "numpy") for mu in mu_syms]
    pressure_func = sp.lambdify(all_args, pressure_sym, "numpy")

    def eval_mu_pressure(rho, vij):
        """
        Parameters
        ----------
        rho : array_like, shape (Ns,)
            Densities in the same order as density_syms
        vij : array_like, shape (Ns, Ns) or flat
            Pair interactions

        Returns
        -------
        mu : ndarray
            Chemical potentials
        P : float
            Pressure
        """
        rho = np.asarray(rho, dtype=float)

        vij = np.asarray(vij, dtype=float)
        if vij.ndim == 2:
            vij = vij.reshape(-1)

        args = np.concatenate([rho, vij])

        mu = np.array([f(*args) for f in mu_funcs])
        P = pressure_func(*args)

        return mu, P

    return {
        "density_symbols": density_syms,
        "vij_symbols": vij_syms,
        "free_energy": f_sym,
        "mu_expressions": mu_syms,
        "pressure_expression": pressure_sym,
        "eval_mu_pressure": eval_mu_pressure,
    }



from pathlib import Path
import numpy as np
import json


def scan_isocore_direct(
    ctx,
    config_dict,
    fe_res,
    supplied_data=None,
    n_points=200,
    output_file="pressure_vs_density.json",
    verbose=True,
):
    """
    Direct density scan:
    - Intrinsic constraint fixes one or more densities.
    - Extrinsic constraint provides scanning density upper bound.
    - No root solving.
    - Just compute mu and pressure.
    """

    thermo = build_thermodynamics_from_fe_res(fe_res)
    eval_mu_pressure = thermo["eval_mu_pressure"]

    species = list(deep_get(config_dict, "species"))
    intrinsic = deep_get(config_dict, "intrinsic_constraints", {})
    extrinsic = deep_get(config_dict, "extrinsic_constraints", {})

    intrinsic_density_dict = intrinsic.get("density", {})
    extrinsic_density_dict = extrinsic.get("density", {})
    total_density_bound = extrinsic.get("total_density_bound", None)

    N = len(species)

    # ---- Validate ----
    if len(extrinsic_density_dict) != 1:
        raise ValueError("Only one scanning density supported")

    scan_species = list(extrinsic_density_dict.keys())[0]
    rho_scan_max = extrinsic_density_dict[scan_species]

    scan_index = species.index(scan_species)

    fixed_indices = []
    fixed_values = {}

    for sp, val in intrinsic_density_dict.items():
        idx = species.index(sp)
        fixed_indices.append(idx)
        fixed_values[idx] = val

    rho_scan_values = np.linspace(1e-8, rho_scan_max, n_points)

    results = []

    # ---------------------------------------------------
    # Local vij function
    # ---------------------------------------------------
    def compute_vij(densities, kernel="uniform"):
        kernel_out = build_strength_kernel(
            ctx=ctx,
            config=config_dict,
            supplied_data=supplied_data,
            densities=densities,
            kernel_type=kernel,
        )

        r = kernel_out["r"]
        kernel = kernel_out["kernel"]

        kernel_dict = {}
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                kernel_dict[(si, sj)] = {"r": r, "values": kernel[i, j]}

        vij_out = vij_radial_kernel(
            ctx=ctx,
            config=config_dict,
            kernel=kernel_dict,
            supplied_data=None,
            export_json=True,
        )

        vij = np.zeros((N, N))
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                vij[i, j] = vij_out["vij_numeric"][(si, sj)]

        return vij

    # ---------------------------------------------------
    # Main scan loop
    # ---------------------------------------------------
    for rho_scan in rho_scan_values:

        rho = np.zeros(N)

        # Insert intrinsic fixed densities
        for idx in fixed_indices:
            rho[idx] = fixed_values[idx]

        # Insert scanned density
        rho[scan_index] = rho_scan

        if total_density_bound is not None:
            if np.sum(rho) > total_density_bound:
                continue

        vij = compute_vij(rho)
        mu, P = eval_mu_pressure(rho, vij)

        results.append({
            "rho_scan": float(rho_scan),
            "rho_full": rho.tolist(),
            "pressure": float(P),
            "mue": mu.tolist(),
        })

        if verbose:
            print(f"{scan_species}={rho_scan:.4f}, P={P:.4f}")

    # ---------------------------------------------------
    # Save to scratch
    # ---------------------------------------------------
    scratch = Path(ctx.scratch_dir)
    output_path = scratch / output_file

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    if verbose:
        print(f"Saved to {output_path}")

    return results
