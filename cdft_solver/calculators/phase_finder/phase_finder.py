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



def scan_isochem_multi(
    ctx,
    config_dict,
    fe_res,
    supplied_data=None,
    n_points=200,
    output_file="pressure_vs_density.json",
    verbose=True,
):
    """
    Generalized density scan with multiple chemical potentials fixed.
    """

    thermo = build_thermodynamics_from_fe_res(fe_res)
    eval_mu_pressure = thermo["eval_mu_pressure"]

    species = list(deep_get(config_dict, "species"))
    intrinsic = deep_get(config_dict, "intrinsic_constraints", {})
    extrinsic = deep_get(config_dict, "extrinsic_constraints", {})

    mu_targets = intrinsic.get("chemical_potential", {})
    fixed_density_dict = extrinsic.get("density", {})
    total_density_bound = extrinsic.get("total_density_bound", None)

    mu_species = list(mu_targets.keys())
    fixed_species = list(fixed_density_dict.keys())
    
    
    def compute_vij(densities, kernel):
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
        for i, si in enumerate(species_names):
            for j, sj in enumerate(species_names):
                kernel_dict[(si, sj)] = {"r": r, "values": kernel[i, j]}

        vij_out = vij_radial_kernel(
            ctx=ctx,
            config=config_dict,
            kernel=kernel_dict,
            supplied_data=None,
            export_json=True,
        )

        vij = np.zeros((N_species, N_species))
        for i, si in enumerate(species_names):
            for j, sj in enumerate(species_names):
                vij[i, j] = vij_out["vij_numeric"][(si, sj)]

        return vij
    
    

    # Unknown species = those not fixed
    unknown_species = [s for s in species if s not in fixed_species]

    if len(mu_species) != len(unknown_species):
        raise ValueError(
            "Number of μ constraints must equal number of unknown densities"
        )

    mu_indices = [species.index(s) for s in mu_species]
    fixed_indices = [species.index(s) for s in fixed_species]
    unknown_indices = [species.index(s) for s in unknown_species]

    # Build scanning grid over fixed density (if one fixed)
    if len(fixed_species) != 1:
        raise ValueError("Scan currently supports one scanning density")

    fixed_sp = fixed_species[0]
    rho_fixed_max = fixed_density_dict[fixed_sp]
    rho_fixed_values = np.linspace(1e-6, rho_fixed_max, n_points)

    results = []
    last_solution = np.ones(len(unknown_species)) * 0.1

    for rho_fixed in rho_fixed_values:

        def root_func(rho_unknown):
            rho = np.zeros(len(species))

            # insert fixed
            rho[fixed_indices[0]] = rho_fixed

            # insert unknown guesses
            for idx, val in zip(unknown_indices, rho_unknown):
                rho[idx] = val

            if total_density_bound is not None:
                if np.sum(rho) > total_density_bound:
                    return np.ones(len(mu_species)) * 1e6

            vij = compute_vij(rho, kernel="uniform")
            print (vij)
            
            
            mu, _ = eval_mu_pressure(rho, vij)

            return np.array([
                mu[i] - mu_targets[species[i]]
                for i in mu_indices
            ])

        try:
            sol = root(root_func, last_solution)
        except:
            continue

        if not sol.success:
            continue

        rho_unknown_solution = sol.x
        last_solution = rho_unknown_solution

        rho = np.zeros(len(species))
        rho[fixed_indices[0]] = rho_fixed
        for idx, val in zip(unknown_indices, rho_unknown_solution):
            rho[idx] = val

        vij = compute_vij(rho, kernel="uniform")
        mu, P = eval_mu_pressure(rho, vij)

        results.append({
            "rho_scan": float(rho_fixed),
            "rho_solution": rho.tolist(),
            "pressure": float(P),
        })

        if verbose:
            print(f"{fixed_sp}={rho_fixed:.4f}, P={P:.4f}")
            
    scratch = Path(ctx.scratch_dir)

    output_file = Path(scratch/output_file)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    return results
