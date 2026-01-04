# coexistence_isochor_fixed.py

import json
import ast
import numpy as np
import sympy as sp
from scipy.optimize import root
import random
from copy import deepcopy
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



# ============================================================
# MAIN SOLVER
# ============================================================
def coexistence_densities_isocore(
    ctx,
    config_dict,
    fe_res,
    supplied_data = None,
    max_outer_iters=10,
    tol_outer=1e-3,
    tol_solver=1e-8,
    verbose=True
):
    """
    Iso-chemical coexistence solver using ONLY a supplied dictionary.
    """

    # --------------------------------------------------------
    # 1) ENSEMBLE CHECK
    # --------------------------------------------------------
    species = deep_get(config_dict, "species")
    ensemble = deep_get(config_dict, "ensemble")
    if ensemble != "isochem":
        raise ValueError(f"Expected ensemble='isochem', got {ensemble}")

    # --------------------------------------------------------
    # 2) FREE ENERGY PARAMETERS
    # --------------------------------------------------------
    integrated_strength_kernel = deep_get(
        config_dict, "integrated_strength_kernel", "uniform"
    ).lower()

    
    # --------------------------------------------------------
    # 3) SOLUTION INITIATOR
    # --------------------------------------------------------
    intrinsic_constraints = deep_get(
        config_dict, "intrinsic_constraints", {}
    )

    extrinsic_constraints = deep_get(
        config_dict, "extrinsic_constraints", {}
    )

    number_of_phases = extrinsic_constraints.get("number_of_phases", 2)
    total_density_bound = extrinsic_constraints.get("total_density_bound", 2.0)
    heterogeneous_pair = extrinsic_constraints.get("heterogeneous_pair", [])

    if isinstance(heterogeneous_pair, str):
        heterogeneous_pair = [heterogeneous_pair]

    if verbose:
        print("[config]")
        print(" ensemble =", ensemble)
        print(" kernel =", integrated_strength_kernel)
        print(" n_phases =", number_of_phases)

    # --------------------------------------------------------
    # 4) TOTAL FREE ENERGY (SYMBOLIC)
    # --------------------------------------------------------
    
    
    if isinstance(heterogeneous_pair, str):
        heterogeneous_pair = [heterogeneous_pair]
   
    # Validate intrinsic constraints count
   

    n_species = N = N_species = len(species)
    species_names = list(species)

    
    # ---------------------------------------------------------------------
    # Species order builder based on heterogeneous pairs
    # ---------------------------------------------------------------------
    # ============================================================
    # Normalize heterogeneous_pair and validate constraints
    # ============================================================

    # --- Normalize heterogeneous_pair ---
    heterogeneous_pair = extrinsic_constraints.get("heterogeneous_pair", [])

    if heterogeneous_pair is None:
        heterogeneous_pair = []
    elif isinstance(heterogeneous_pair, (str, int)):
        heterogeneous_pair = [str(heterogeneous_pair)]
    elif isinstance(heterogeneous_pair, (list, tuple)):
        heterogeneous_pair = [str(p) for p in heterogeneous_pair]
    else:
        raise TypeError(
            f"heterogeneous_pair must be str or list-like, got {type(heterogeneous_pair)}"
        )


    # ============================================================
    # Validate intrinsic constraints count
    # ============================================================

    def count_intrinsic_constraints(constraints):
        """
        Counts independent intrinsic constraints in a robust way.
        Supports:
          - scalar constraints
          - per-species dict constraints
        """
        count = 0
        for _, val in constraints.items():
            if isinstance(val, dict):
                count += len(val)
            else:
                count += 1
        return count


    species_names = list(species)
    N_species = len(species_names)


    # ============================================================
    # Diagnostics
    # ============================================================

    if verbose:
        print("  Loaded coexistence configuration:")
        print(f"    number_of_phases     = {number_of_phases}")
        print(f"    heterogeneous_pair  = {heterogeneous_pair}")
        print(f"    total_density_bound = {total_density_bound}")
        print(f"    intrinsic_constraints = {intrinsic_constraints}")
        print(f"    extrinsic_constraints = {extrinsic_constraints}")


    pvec = []  # list to hold species fraction values
    if "species_fraction" in intrinsic_constraints:
        sf = intrinsic_constraints["species_fraction"]
        if isinstance(sf, dict):
            # Extract only the numeric values in order of species name sorting
            pvec = [sf[sp] for sp in species_names]
        else:
            raise ValueError("❌ 'species_fraction' should be a dictionary of species: value pairs.")

            # --- Load extrinsic constraints ---
    if "species_fraction" in intrinsic_constraints:
        print(f"  species_fraction values (pvec) = {pvec}")
    else:
        print(f"  intrinsic_constraints = {intrinsic_constraints}")



    # ============================================================
    # Species ordering logic (heterogeneous-pair aware)
    # ============================================================
    
    # --------------------------------------------------------
    # 6) REDUCED VARIABLES
    # --------------------------------------------------------
    def reduced_to_densities(block):
        rhot = block[0]
        fracs = block[1:]
        rho = []
        prod = 1.0
        for x in fracs:
            prod *= (1 - x)
        factor = 1.0
        for i in range(len(fracs) + 1):
            rho.append(rhot * prod * factor)
            if i < len(fracs):
                prod /= (1 - fracs[i])
                factor = fracs[i]
        return rho

    # --------------------------------------------------------
    # 7) vᵢⱼ COMPUTATION (UNIFIED)
    # --------------------------------------------------------
    

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


    # --------------------------------------------------------
    # 8) RESIDUAL
    # --------------------------------------------------------
    # ============================================================
    # Coexistence residual (dictionary-driven, framework-safe)
    # ============================================================
    
    thermo = build_thermodynamics_from_fe_res(fe_res)

    eval_mue_pressure_fn = thermo["eval_mu_pressure"]


   
    def coexistence_residual_isocore(
        vars_vec,
        n_phases,
        species_names,
        pvec,
        eval_mue_pressure_fn,
    ):
        """
        Residual for ISOCHORE coexistence with fixed total species densities pvec.

        Unknowns:
          - For each phase: [rho_total, x1, x2, ..., x_(N-1)]
          - Phase fractions: (n_phases - 1)

        Constraints:
          - μ_i equal across phases
          - Pressure equal across phases
          - Σ_p φ_p ρ_i^(p) = pvec[i]   (isochore constraint)
        """

        global CURRENT_VIJ_PER_PHASE
        if CURRENT_VIJ_PER_PHASE is None:
            raise RuntimeError("CURRENT_VIJ_PER_PHASE is not initialized")

        N = len(species_names)
        reduced_len = N  # rhot + (N-1) composition variables

        # -------------------------------------------------
        # Split reduced variables per phase
        # -------------------------------------------------
        reduced_blocks = []
        idx = 0
        for _ in range(n_phases):
            reduced_blocks.append(vars_vec[idx:idx + reduced_len].tolist())
            idx += reduced_len

        # -------------------------------------------------
        # Phase fractions
        # -------------------------------------------------
        frac_vars = vars_vec[idx:]
        if len(frac_vars) != (n_phases - 1):
            raise ValueError("Incorrect number of phase-fraction variables")

        fractions = list(frac_vars)
        fractions.append(1.0 - sum(fractions))

        # -------------------------------------------------
        # Recover species densities per phase
        # -------------------------------------------------
        rhos_per_phase = [
            reduced_to_densities(block) for block in reduced_blocks
        ]

        # -------------------------------------------------
        # Evaluate μ and P
        # -------------------------------------------------
        mu_vals = []
        pressure_vals = []

        for p in range(n_phases):
            mu_p, p_p = eval_mue_pressure_fn(
                rhos_per_phase[p],
                CURRENT_VIJ_PER_PHASE[p]
            )
            mu_vals.append(mu_p)
            pressure_vals.append(p_p)

        # -------------------------------------------------
        # Build residual equations
        # -------------------------------------------------
        eqs = []

        # (1) Chemical potential equalities
        for i in range(N):
            for p in range(n_phases - 1):
                eqs.append(mu_vals[p][i] - mu_vals[p + 1][i])

        # (2) Pressure equalities
        for p in range(n_phases - 1):
            eqs.append(pressure_vals[p] - pressure_vals[p + 1])

        # (3) Isochore mass conservation (FIXED TOTAL DENSITIES)
        for i in range(N):
            lhs = sum(
                fractions[p] * rhos_per_phase[p][i]
                for p in range(n_phases)
            )
            eqs.append(lhs - pvec[i])

        return np.asarray(eqs, dtype=float)



    # ============================================================
    # Random initial guess generator
    # ============================================================
    def random_initial_guess(n_phases, n_species, total_density_bound):
        eps_rho = 1e-3
        eps_x   = 1e-4

        N = n_species
        M = N - 1
        guess = []

        for _ in range(n_phases):
            guess.append(np.random.uniform(eps_rho, total_density_bound))
            for _ in range(M):
                guess.append(np.random.uniform(0, 1.0))

        remaining = 1.0
        for _ in range(n_phases - 1):
            val = np.random.uniform(0.0, remaining)
            guess.append(val)
            remaining = max(remaining - val, eps_rho)
            
        print(len(guess))

        return np.asarray(guess)

    
    def get_species_order(species_names, heterogeneous_pairs):
        """
        Builds a stable species ordering based on heterogeneous-pair constraints.

        heterogeneous_pairs:
            list of strings like ["ab", "ac"] meaning species a and b
            should be adjacent in ordering.
        """
        ordered = list(species_names)

        for pair in heterogeneous_pairs:
            pair = pair.strip()
            if len(pair) != 2:
                continue

            s1, s2 = pair[0], pair[1]
            if s1 in ordered and s2 in ordered:
                ordered.remove(s2)
                idx = ordered.index(s1)
                ordered.insert(idx + 1, s2)

        reshuffle_back = [species_names.index(s) for s in ordered]
        restore_original = [ordered.index(s) for s in species_names]

        return ordered, reshuffle_back, restore_original


    def reorder_to_original_order(rho, restore_original):
        """Reorder densities back to original species order."""
        return [rho[i] for i in restore_original]


    # ============================================================
    # Build ordering maps
    # ============================================================

    species_ordered, reshuffle_back, restore_original = get_species_order(
        species_names, heterogeneous_pair
    )

    
    
    print(pvec)

    # ============================================================
    # General coexistence solver
    # ============================================================

    def solve_coexistence_isocore(
        n_phases,
        species_names,
        pvec,
        eval_mue_pressure_fn,
        heterogeneous_pair,
        total_density_bound,
        max_attempts=200000,
        verbose=True,
    ):
        """
        Solves isochore coexistence for fixed total species densities pvec.
        """

        N = len(species_names)

        # --------------------------
        # Parse heterogeneous pairs
        # --------------------------
        hetero_pairs = []
        for pair in heterogeneous_pair:
            if len(pair) == 2:
                hetero_pairs.append((pair[0], pair[1]))

        if verbose:
            print(
                f"[isochore] Solving for {n_phases} phases, "
                f"{N} species, fixed totals pvec={pvec}"
            )

        for attempt in range(max_attempts):
        
        
            #print (attempt)
            guess = random_initial_guess(
                n_phases, total_density_bound, N
            )

            
            sol = root(
                lambda v: coexistence_residual_isocore(
                    v,
                    n_phases,
                    species_names,
                    pvec,
                    eval_mue_pressure_fn,
                ),
                guess,
                method="hybr",
            )
            

            if not sol.success:
                if verbose and attempt % 2000 == 0:
                    print(f"[attempt {attempt}] solver not converged")
                continue
            # --------------------------------------------------
            # Decode solution
            # --------------------------------------------------
            vars_sol = sol.x
            reduced_len = N

            reduced_blocks = []
            idx = 0
            for _ in range(n_phases):
                reduced_blocks.append(vars_sol[idx:idx + reduced_len].tolist())
                idx += reduced_len

            frac_vars = vars_sol[idx:]
            fractions = list(frac_vars)
            fractions.append(1.0 - sum(fractions))

            rhos_per_phase = [
                reduced_to_densities(block) for block in reduced_blocks
            ]

            # --------------------------------------------------
            # Heterogeneous-pair dominance check
            # --------------------------------------------------
            dominant_threshold = 0.9
            species_max = {
                sp: max(rhos_per_phase[p][i] for p in range(n_phases))
                for i, sp in enumerate(species_names)
            }

            hetero_ok = True
            for p in range(n_phases):
                dominant_species = [
                    sp for i, sp in enumerate(species_names)
                    if rhos_per_phase[p][i] / (species_max[sp] + 1e-14)
                    >= dominant_threshold
                ]
                for s1, s2 in hetero_pairs:
                    if s1 in dominant_species and s2 in dominant_species:
                        hetero_ok = False
                        break
                if not hetero_ok:
                    break

            if not hetero_ok:
                continue

            # --------------------------------------------------
            # Bounds check
            # --------------------------------------------------
            if any(
                rho < 0.0 or rho > total_density_bound
                for phase in rhos_per_phase
                for rho in phase
            ):
                continue

            # --------------------------------------------------
            # Evaluate μ and P
            # --------------------------------------------------
            mu_vals = []
            pressure_vals = []

            global CURRENT_VIJ_PER_PHASE
            for p in range(n_phases):
                mu_p, p_p = eval_mue_pressure_fn(
                    rhos_per_phase[p],
                    CURRENT_VIJ_PER_PHASE[p],
                )
                mu_vals.append([float(x) for x in mu_p])
                pressure_vals.append(float(p_p))

            return {
                "rhos_per_phase": rhos_per_phase,
                "fractions": fractions,
                "mu_per_phase": mu_vals,
                "pressure_per_phase": pressure_vals,
            }

        if verbose:
            print("❌ No valid isochore coexistence solution found")
        return None



    # ============================================================
    # Outer integrated-strength (v_ij) iteration loop
    # ============================================================

    n_phases = number_of_phases
    species_names = list(species)

    # Initial density guess per phase
    initial_rhos = [np.ones(N) * 0.1 for _ in range(n_phases)]

    vij_per_phase = [
        compute_vij(initial_rhos[p], kernel="uniform")
        for p in range(n_phases)
    ]
    
    

    global CURRENT_VIJ_PER_PHASE
    CURRENT_VIJ_PER_PHASE = vij_per_phase

    final_solution = None

    for outer_iter in range(1, max_outer_iters + 1):
        if verbose:
            print(f"[outer] iteration {outer_iter}")

        CURRENT_VIJ_PER_PHASE = vij_per_phase

        sol = solve_coexistence_isocore(
            n_phases=n_phases,
            species_names=species_names,
            pvec=pvec,
            eval_mue_pressure_fn=eval_mue_pressure_fn,
            heterogeneous_pair=heterogeneous_pair,
            total_density_bound=total_density_bound,
            verbose=verbose,
        )

        if sol is None:
            raise RuntimeError("Isochore coexistence solver failed")

        rhos_per_phase = sol["rhos_per_phase"]

        # --------------------------
        # Convergence check
        # --------------------------
        if outer_iter > 1:
            drho = np.concatenate([
                np.asarray(rhos_per_phase[p]) -
                np.asarray(last_rhos_per_phase[p])
                for p in range(n_phases)
            ])
            if np.linalg.norm(drho) < tol_outer:
                if verbose:
                    print("[outer] converged")
                final_solution = sol
                break

        # Update vij
        vij_per_phase = [
            compute_vij(rhos_per_phase[p], kernel=integrated_strength_kernel)
            for p in range(n_phases)
        ]

        last_rhos_per_phase = [np.asarray(r).copy() for r in rhos_per_phase]
        final_solution = sol


    # ============================================================
    # Final output dictionary
    # ============================================================

    if final_solution is None:
        raise RuntimeError("No converged coexistence solution")

    out = {
        "ensemble": "isochore",
        "species": species_names,
        "n_phases": n_phases,
        "rhos_per_phase": final_solution["rhos_per_phase"],
        "mu_per_phase": final_solution["mu_per_phase"],
        "pressure_per_phase": final_solution["pressure_per_phase"],
        "vij_per_phase": [v.tolist() for v in vij_per_phase],
    }

    return out
