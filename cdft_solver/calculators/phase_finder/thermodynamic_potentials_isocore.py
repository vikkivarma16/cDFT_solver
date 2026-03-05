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


def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None


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
    f_func = sp.lambdify(all_args, f_sym, "numpy")

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
        f = f_func(*args)

        return mu, P, f

    return {
        "density_symbols": density_syms,
        "vij_symbols": vij_syms,
        "free_energy": f_sym,
        "mu_expressions": mu_syms,
        "pressure_expression": pressure_sym,
        "eval_mu_pressure": eval_mu_pressure,
    }



    
    
def evaluate_canonical_state(
    ctx,
    config_dict,
    fe_res,
    supplied_data=None,
    export=True,
    output_file="state_from_density.json",
    verbose=True,
):
    """
    Canonical evaluation:
    Input: densities of all species (extrinsic constraint)
    Output: chemical potentials and pressure
    """

    thermo = build_thermodynamics_from_fe_res(fe_res)
    eval_mu_pressure = thermo["eval_mu_pressure"]

    species = list(deep_get(config_dict, "species"))
    extrinsic = deep_get(config_dict, "extrinsic_constraints", {})

    density_dict = extrinsic.get("density", {})

    if len(density_dict) != len(species):
        raise ValueError("You must supply densities for ALL species.")

    # Build density vector in correct order
    rho = np.array([density_dict[s] for s in species])
    
    
    
    

    # Compute interaction matrix
    def compute_vij(densities, kernel):
        kernel_out = build_strength_kernel(
            ctx=ctx,
            config=config_dict,
            supplied_data=supplied_data,
            densities=densities,
            kernel_type=kernel,
        )
        r = kernel_out["r"]
        kernel_vals = kernel_out["kernel"]

        kernel_dict = {}
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                kernel_dict[(si, sj)] = {"r": r, "values": kernel_vals[i, j]}

        vij_out = vij_radial_kernel(
            ctx=ctx,
            config=config_dict,
            kernel=kernel_dict,
            supplied_data=None,
            export_json=False,
        )

        vij = np.zeros((len(species), len(species)))
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                vij[i, j] = vij_out["vij_numeric"][(si, sj)]

        return vij

    free_energy  = find_key_recursive(config_dict, free_energy)
    integrated_strength_kernel = free_energy["integrated_strength_kernel"]
    vij = compute_vij(rho, kernel=integrated_strength_kernel)
    
    # Evaluate thermodynamics
    mu, P, f = eval_mu_pressure(rho, vij)

    result = {
        "ensemble": "canonical",
        "species": species,
        "densities": rho.tolist(),
        "chemical_potentials": mu.tolist(),
        "pressure": float(P),
        "free_energy_density": float(f),
    }

    if verbose:
        print("Densities:", rho)
        print("Chemical potentials:", mu)
        print("Pressure:", P)

    if export:
        scratch = Path(ctx.scratch_dir)
        output_file = scratch / output_file
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)

    return result
