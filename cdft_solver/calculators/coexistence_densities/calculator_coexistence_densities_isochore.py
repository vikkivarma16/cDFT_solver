# coexistence_isochore_fixed.py
import json
import ast
import numpy as np
import sympy as sp
from pathlib import Path
from scipy.optimize import root
import random

# Global used by residual/evaluation helpers to access current vij per phase
CURRENT_VIJ_PER_PHASE = None


def coexistence_densities_isochore(ctx, max_outer_iters=10, tol_outer=1e-3,
                                  tol_solver=1e-8, verbose=True):
    """
    Iso-chemical coexistence solver:
      - builds symbolic μ and pressure from total_free_energy(ctx) output
      - lambdifies μ and pressure as functions of densities and v_ij
      - runs the structured coexistence solver (intrinsic constraints handling)
      - iteratively updates integrated-strength kernel (v_k) according to
        integrated_strength_kernel and supplied_data flags in
        scratch/input_data_free_energy_parameters.json

    Expects total_free_energy(ctx) to return a dict (fe_res) containing keys:
      - 'species', 'densities' (or 'densities_symbols'), 'vij', 'free_energy_symbolic', ...
    """

    # --------------------------
    # dynamic imports from your solver (kept as-is)
    # --------------------------
    from cdft_solver.calculators.integrated_strength_uniform.calculator_integrated_strength_uniform import vk_uniform
    from cdft_solver.calculators.integrated_strength_void.calculator_integrated_strength_void_supplied_data import vk_void_supplied_data
    from cdft_solver.calculators.integrated_strength_rdf.calculator_integrated_strength_rdf import vk_rdf
    from cdft_solver.calculators.integrated_strength_rdf.calculator_integrated_strength_rdf_supplied_data import vk_rdf_supplied_data
    from cdft_solver.calculators.total_free_energy.calculator_total_free_energy import total_free_energy

    # --------------------------
    # 1) read ensemble from input file (must be isochore)
    # --------------------------
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)

    
    
    ensemble = None
    json_file_solution_initiator = Path(scratch) / "input_data_solution_initiator.json"

    try:
        with open(json_file_solution_initiator, "r") as file:
            data = json.load(file)
            ensemble = data.get("ensemble", None)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Could not find solution initiator file: {json_file_solution_initiator}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to parse JSON from {json_file_solution_initiator}: {e}")

    if ensemble != "isochores":
        raise ValueError(f"❌ Expected ensemble = 'isochores', but got '{ensemble}'")

    
    
    

    # --------------------------
    # 2) load vk config (which kernel to use and whether supplied_data yes/no)
    # --------------------------
    vk_json_file = scratch / "input_data_free_energy_parameters.json"
    if not vk_json_file.exists():
        raise FileNotFoundError(f"Missing file: {vk_json_file}")
    with open(vk_json_file, "r") as fh:
        vk_params = json.load(fh).get("free_energy_parameters", {})

    integrated_strength_kernel = vk_params.get("integrated_strength_kernel", "uniform").lower()
    supplied_data_flag = vk_params.get("supplied_data", "no").lower()

    # map kernel -> internal modes used by compute_vij_for_mode
    if integrated_strength_kernel == "uniform":
        vk_mode_initial = "uniform"
        vk_mode_iter = "uniform"
    elif integrated_strength_kernel == "void":
        vk_mode_initial = "uniform"
        vk_mode_iter = "void_supplied" if supplied_data_flag == "yes" else "void"
    elif integrated_strength_kernel == "rdf":
        vk_mode_initial = "uniform"
        vk_mode_iter = "rdf_supplied" if supplied_data_flag == "yes" else "rdf"
    else:
        raise ValueError(f"Unknown integrated_strength_kernel: {integrated_strength_kernel}")

    if verbose:
        print(f"[isochore] VK modes: initial='{vk_mode_initial}', iter='{vk_mode_iter}'")

    # --------------------------
    # 3) load solution initiator (intrinsic/extrinsic constraints)
    # --------------------------
    
    
    
    # =====================================================================
    # --- SOLUTION SECTION (consistent and reorder-safe) ---
    # =====================================================================

    # --- Load JSON file with solution initiator parameters ---
    json_file_solution_initiator = Path(scratch) / "input_data_solution_initiator.json"

    # --- Default values ---
    number_of_phases = 2
    heterogeneous_pair = ['ab']
    total_density_bound = 2.0
    intrinsic_constraints = {}
    extrinsic_constraints = {}
    pvec = []  # list to hold species fraction values


   
    try:
        with open(json_file_solution_initiator, "r") as file:
            data_solution = json.load(file)["solution_initiator"]
            print (data_solution)
            
            # --- Load intrinsic constraints (chemical potentials, pressure, or species fractions) ---
            intrinsic_constraints = data_solution.get("intrinsic_constraints", {})


            
            # --- Handle species_fraction explicitly ---
            if "species_fraction" in intrinsic_constraints:
                sf = intrinsic_constraints["species_fraction"]
                if isinstance(sf, dict):
                    # Extract only the numeric values in order of species name sorting
                    pvec = [sf[key] for key in sorted(sf.keys())]
                else:
                    raise ValueError("❌ 'species_fraction' should be a dictionary of species: value pairs.")

            # --- Load extrinsic constraints ---
            extrinsic_constraints = data_solution.get("extrinsic_constraints", {})
            number_of_phases = extrinsic_constraints.get("number_of_phases", number_of_phases)
            heterogeneous_pair = extrinsic_constraints.get("heterogeneous_pair", heterogeneous_pair)
            total_density_bound = extrinsic_constraints.get("total_density_bound", total_density_bound)

            # Ensure heterogeneous_pair is always a list
            if isinstance(heterogeneous_pair, str):
                heterogeneous_pair = [heterogeneous_pair]

    except FileNotFoundError:
        print(f"⚠️ JSON file not found: {json_file_solution_initiator}")
    except KeyError as e:
        print(f"⚠️ Missing expected key in JSON: {e}")
    except json.JSONDecodeError:
        print(f"⚠️ Failed to decode JSON file: {json_file_solution_initiator}")
    except ValueError as e:
        print(f"⚠️ Invalid data format in JSON: {e}")

    # --- Summary output ---
    print(f"Loaded solution initiator parameters:")
    print(f"  number_of_phases = {number_of_phases}")
    print(f"  heterogeneous_pair = {heterogeneous_pair}")
    print(f"  total_density_bound = {total_density_bound}")
    if "species_fraction" in intrinsic_constraints:
        print(f"  species_fraction values (pvec) = {pvec}")
    else:
        print(f"  intrinsic_constraints = {intrinsic_constraints}")
    
    

    # --------------------------
    # 4) Obtain free energy symbolic result and parse fields (handle stringified lists)
    # --------------------------
    fe_res = total_free_energy(ctx)
    if fe_res is None:
        raise RuntimeError("total_free_energy(ctx) returned None")

    def _maybe_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
        return x

    species = _maybe_eval(fe_res.get("species"))
    if species is None:
        raise RuntimeError("free energy result missing 'species'")
    species_names =  species

    densities_names = _maybe_eval(fe_res.get("densities"))
    if densities_names is None:
        densities_names = _maybe_eval(fe_res.get("densities_symbols"))
        if densities_names is None:
            raise RuntimeError("free energy result missing 'densities' or 'densities_symbols'")

    vij_raw = _maybe_eval(fe_res.get("vij"))
    volume_factors = _maybe_eval(fe_res.get("volume_factors", []))
    f_symbolic_raw = fe_res.get("free_energy_symbolic")
    if f_symbolic_raw is None:
        raise RuntimeError("free energy result missing 'free_energy_symbolic'")

    if isinstance(f_symbolic_raw, str):
        f_sym = sp.sympify(f_symbolic_raw)
    elif isinstance(f_symbolic_raw, sp.Expr):
        f_sym = f_symbolic_raw
    else:
        f_sym = sp.sympify(str(f_symbolic_raw))

    # --------------------------
    # 5) build sympy symbols for densities and vij
    # --------------------------
    density_syms = [sp.Symbol(name) if not isinstance(name, sp.Expr) else name for name in densities_names]


    N = len(density_syms)

    if vij_raw is None:
        raise RuntimeError("vij specification missing")
    vij_nested = vij_raw
    if isinstance(vij_nested[0], str):
        raise RuntimeError("vij must be nested list of shape (N,N)")
    vij_flat_names = [s for row in vij_nested for s in row]
    vij_syms = [sp.Symbol(name) for name in vij_flat_names]

    sym_map = {}
    for s in density_syms:
        sym_map[str(s)] = s
    for s in vij_syms:
        sym_map[str(s)] = s

    replacements = {}
    for token in f_sym.free_symbols:
        name = str(token)
        if name in sym_map:
            replacements[token] = sym_map[name]

    if replacements:
        f_sym = f_sym.xreplace(replacements)
        f_sym = sp.sympify(f_sym)
        
    all_args = density_syms + vij_syms
    f_funcs  = sp.lambdify(all_args, f_sym, "numpy") 


    # --------------------------
    # 6) Build chemical potentials and pressure (symbolic)
    # --------------------------
    mue_syms = [sp.diff(f_sym, density_syms[i]) for i in range(N)]
    pressure_sym = -f_sym + sum(density_syms[i] * mue_syms[i] for i in range(N))
    

    # --------------------------
    # 7) Lambdify mue and pressure: argument order densities... then vij_flat...
    # --------------------------
    all_args = density_syms + vij_syms
    mue_funcs = [sp.lambdify(all_args, mu_expr, "numpy") for mu_expr in mue_syms]
    
    pressure_func = sp.lambdify(all_args, pressure_sym, "numpy")

    def eval_mue_pressure(rho_array, vij_matrix):
        rho_arr = np.asarray(rho_array, dtype=float).reshape(-1)
        vij_arr = np.asarray(vij_matrix, dtype=float)
        if vij_arr.shape != (N, N):
            raise ValueError(f"vij_matrix must be shape ({N},{N}), got {vij_arr.shape}")
        vij_flat = vij_arr.reshape(-1)
        args = tuple(np.concatenate([rho_arr, vij_flat]).tolist())
        mue_vals = np.array([f(*args) for f in mue_funcs], dtype=float)
        p_val = float(pressure_func(*args))
        return mue_vals, p_val

    # --------------------------
    # 8) Helper: reduced <-> full densities
    # --------------------------
    def reduced_to_densities(reduced_vars):
        """
        Convert reduced variables to full species densities for one phase.

        reduced_vars = [rhot, x1, x2, ..., xM], with M = N-1
        returns: rho = [rho_1, rho_2, ..., rho_N]
        """
        rhot = reduced_vars[0]
        fractions = reduced_vars[1:]

        rho = []
        prod = 1.0
        for x in fractions:
            prod *= (1.0 - x)

        factor = 1.0
        for i in range(len(fractions) + 1):
            val = rhot * prod * factor
            rho.append(val)
            if i < len(fractions):
                prod *= 1.0 / (1.0 - fractions[i])
                factor = fractions[i]
        return rho

   
   
    def compute_vij_for_mode(densities_phase, mode):
        """
        densities_phase: array-like (numpy array or list) of length N
        mode: one of "uniform", "void", "rdf", "rdf_supplied", "void_supplied"
        """
        # normalize densities to a plain Python list (safe for JSON/APIs)
        try:
            densities_clean = np.array(densities_phase).astype(float).tolist()
        except Exception as e:
            raise ValueError(f"Invalid densities_phase provided: {densities_phase!r}") from e

        if mode == "uniform":
            return np.array(vk_uniform(ctx)["vk"])
        elif mode == "void":
            return np.array(vk_void_supplied_data(ctx)["vk"])
        elif mode == "rdf":
            # pass densities as keyword arg so it maps to the correct parameter
            try:
                print("Calling vk_rdf with densities =", densities_clean)
                return np.array(vk_rdf(ctx, densities=densities_clean)["vk"])
            except TypeError as e:
                # defensive: some older signatures might differ — try positional with beta first
                try:
                    print("TypeError calling vk_rdf(...). Trying positional with beta then densities.")
                    return np.array(vk_rdf(ctx, 1.0, densities_clean)["vk"])
                except Exception as e2:
                    print("vk_rdf failed:", e2)
                    raise
            except Exception as e:
                print("vk_rdf failed:", e)
                raise
        elif mode == "rdf_supplied":
            return np.array(vk_rdf_supplied_data(ctx)["vk"])
        elif mode == "void_supplied":
            return np.array(vk_void_supplied_data(ctx)["vk"])
        else:
            raise ValueError(f"Unknown vk mode '{mode}'")

    
    
    
    # ---------------------------------------------------------------------
    # Species order builder based on heterogeneous pairs
    # ---------------------------------------------------------------------
    def get_species_order(species_names, heterogeneous_pair):
        """
        Builds a consistent ordering of species based on heterogeneous pairs.
        heterogeneous_pair can be a string ('ab,ac') or a list (['ab', 'ac']).
        """
        # Normalize to list of strings
        if isinstance(heterogeneous_pair, str):
            hetero_pairs = [pair.strip() for pair in heterogeneous_pair.split(',') if pair.strip()]
        elif isinstance(heterogeneous_pair, list):
            hetero_pairs = [str(pair).strip() for pair in heterogeneous_pair if str(pair).strip()]
        else:
            raise ValueError(f"Unexpected type for heterogeneous_pair: {type(heterogeneous_pair)}")

        # Start with original order
        ordered = list(species_names)

        for pair in hetero_pairs:
            if len(pair) == 2:
                i1, i2 = pair[0], pair[1]
                if i1 in ordered and i2 in ordered:
                    ordered.remove(i2)
                    idx = ordered.index(i1)
                    ordered.insert(idx + 1, i2)

        # avoid shadowing 'sp' etc — use explicit variable names
        reshuffle_back = [species_names.index(species_name) for species_name in ordered]
        restore_original = [ordered.index(species_name) for species_name in species_names]

        return ordered, reshuffle_back, restore_original


    def reorder_to_original_order(rho, restore_original):
        """Reorder computed densities to match original species order."""
        return [rho[i] for i in restore_original]


    # Build species order maps
    species_ordered, reshuffle_back, restore_original = get_species_order(species_names, heterogeneous_pair)

    
    

    # coexistence residual using CURRENT_VIJ_PER_PHASE
    def coexistence_residual_general(vars, n_phases, sigmaij, mue_value_funcs, pressure_value_func, pvec):
    
        Nlocal = len(sigmaij)
        M = Nlocal - 1
        per_phase_len = M + 1
        reduced_blocks = []
        
        idx = 0
        for _ in range(n_phases):
            reduced_blocks.append(vars[idx: idx + per_phase_len].tolist())
            idx += per_phase_len
            
        new_rhos_per_phase = [reduced_to_densities(block) for block in reduced_blocks]
        rhos_per_phase = [reorder_to_original_order(rho, restore_original) for rho in new_rhos_per_phase]
        
        mu_per_phase = []
        pressure_per_phase = []
        global CURRENT_VIJ_PER_PHASE
        if CURRENT_VIJ_PER_PHASE is None:
            raise RuntimeError("CURRENT_VIJ_PER_PHASE must be set before calling the solver")
            
        
        frac_vars = vars[idx:]
        if len(frac_vars) != (n_phases - 1):
            raise ValueError("Incorrect number of phase fraction variables in residual.")
        fractions = list(frac_vars)
        fractions.append(1.0 - sum(fractions))
           
            
            
        for p in range(n_phases):
            rho_p = new_rhos_per_phase[p]
            vij_p = CURRENT_VIJ_PER_PHASE[p]
            mu_p, p_p = eval_mue_pressure(rho_p, vij_p)
            mu_per_phase.append(mu_p)
            pressure_per_phase.append(p_p)
        eqs = []

        # (1) Chemical potential equalities
        for i in range(N):
            for p in range(n_phases - 1):
                eqs.append(mu_per_phase[p][i] - mu_per_phase[p + 1][i])

        # (2) Pressure equalities
        for p in range(n_phases - 1):
            eqs.append(pressure_per_phase[p] - pressure_per_phase[p + 1])

        # (3) Mass conservation
        for i in range(N):
            lhs = sum(fractions[p] * rhos_per_phase[p][i] for p in range(n_phases))
            eqs.append(lhs - pvec[i])

        return np.array(eqs)


    # ---------------------------------------------------------------------
    # Random initial guess generator (preserving reordering)
    # ---------------------------------------------------------------------
    def random_initial_guess(n_phases, sigmaij):
        """
        Generates an initial guess vector:
          n_phases * (M+1) reduced vars + (n_phases - 1) fractions
        Preserves reorder consistency.
        """
        N = len(sigmaij)
        M = N - 1
        per_phase = M + 1
        guess = []

        for phase_idx in range(n_phases):
            guess.append(np.random.uniform(0.0, total_density_bound))  # rhot
            for i in range(M):
                base = (phase_idx / n_phases)
                val = np.random.uniform(0.0, 1.0)
                guess.append(val)

        # Fractions
        remaining = 1.0
        for k in range(n_phases - 1):
            val = np.random.uniform(0.0, remaining)
            guess.append(val)
            remaining -= val
            if remaining <= 1e-6:
                remaining = 1e-6

        # Preserve reorder structure
        return guess

    def solve_general_coexistence(n_phases, sigmaij, mue_value_funcs, pressure_value_func, pvec, n_attempts=500000, verbose=True, scratch=None):
        N = len(sigmaij)
        per_phase_len = N  # each phase has N densities
        if verbose:
            print(f"Solving coexistence for {n_phases} phases (N={N}, total vars={n_phases*per_phase_len})")

        # Parse heterogeneous pairs if defined globally
        pair_list = []
        for item in heterogeneous_pair:
            parts = [p.strip() for p in item.split(',')]
            for p in parts:
                if len(p) == 2:
                    pair_list.append((p[0], p[1]))
                else:
                    pair_list.append(tuple(p.split()))
        hetero_pairs_parsed = pair_list
        
        for attempt in range(n_attempts):
            guess = random_initial_guess(n_phases, sigmaij)  # should return list of length n_phases*N
            try:
                sol = root(
                    lambda v: coexistence_residual_general(
                        v, n_phases, sigmaij, mue_value_funcs,
                        pressure_value_func, pvec
                    ),
                    guess,
                    method='hybr'
                )
            except Exception as e:
                if verbose:
                    print(f"Attempt {attempt}: solver raised exception: {e}")
                continue

            if not sol.success:
                if verbose and attempt % 2000 == 0:
                    print(f"Attempt {attempt}: solver failed ({sol.message})")
                continue

            vars_sol = sol.x
            reduced_blocks = []
            idx = 0
            for _ in range(n_phases):
                reduced_blocks.append(vars_sol[idx: idx + per_phase_len].tolist())
                idx += per_phase_len

            # Convert & reorder reduced -> full densities
            new_rhos_per_phase = [reduced_to_densities(block) for block in reduced_blocks]
            rhos_per_phase = [reorder_to_original_order(rho, restore_original) for rho in new_rhos_per_phase]

            # --- HETEROGENEOUS-PAIR CHECK ---
            dominant_threshold = 0.9
            hetero_ok = True
            species_max = {sp_name: max(rhos_per_phase[p][i] for p in range(n_phases))
                           for i, sp_name in enumerate(species_names)}
            for p in range(n_phases):
                dominant_species = [sp_name for i, sp_name in enumerate(species_names)
                                    if rhos_per_phase[p][i] / (species_max[sp_name] + 1e-16) >= dominant_threshold]
                for (s1, s2) in hetero_pairs_parsed:
                    if s1 in dominant_species and s2 in dominant_species:
                        hetero_ok = False
                        if verbose:
                            print(f"Attempt {attempt}: rejected by heterogeneous constraint — "
                                  f"both '{s1}' and '{s2}' dominant in phase {p+1}, indicating about a non-disjoint solution.")
                        break
                if not hetero_ok:
                    break
            if not hetero_ok:
                continue

            # --- Bounds check ---
            if any(not (0.0 <= dens <= extrinsic_constraints.get("total_density_bound", 2.0))
                   for block in reduced_blocks for dens in block):
                continue

            # evaluate mu and pressure
            mu_per_phase = []
            pressure_per_phase = []
  
            try:
                global CURRENT_VIJ_PER_PHASE
                if CURRENT_VIJ_PER_PHASE is None:
                    raise RuntimeError("CURRENT_VIJ_PER_PHASE is not set")
                for p in range(n_phases):
                    vij_p = CURRENT_VIJ_PER_PHASE[p]
                    mu_p, p_p = eval_mue_pressure(rhos_per_phase[p], vij_p)
                    mu_per_phase.append([float(x) for x in np.array(mu_p).reshape(-1)])
                    pressure_per_phase.append(float(p_p))
            except Exception as e:
                if verbose:
                    print(f"Attempt {attempt}: failed computing mu/pressure: {e}")
                continue
           
            result = {"species": {}, "pressure": {}}
            for idx_sp, name in enumerate(species_names):
                species_entry = {}
                for p_idx in range(n_phases):
                    species_entry[f"rho_phase_{p_idx+1}"] = float(rhos_per_phase[p_idx][idx_sp])
                    species_entry[f"mue_phase_{p_idx+1}"] = float(mu_per_phase[p_idx][idx_sp])
                result["species"][name] = species_entry
            for p_idx in range(n_phases):
                result["pressure"][f"pressure_phase_{p_idx+1}"] = float(pressure_per_phase[p_idx])
            # export Solution.json
            if scratch is None:
                scratch = Path(".")
            output_file = Path(scratch) / "Solution.json"
            try:
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=4)
                if verbose:
                    print("✅ Coexistence solution exported successfully:")
                    print(f"   → {output_file}")
            except Exception:
                if verbose:
                    print("⚠️ Warning: failed to write Solution.json")
            return {
                "rhos_per_phase": rhos_per_phase,
                "mu_per_phase": mu_per_phase,
                "pressure_per_phase": pressure_per_phase,
                "exported_result": result
            }
            print(attempt)
        if verbose:
            print("❌ No valid coexistence solution found after multiple attempts.")
        return None

    # --------------------------
    # 12) Outer vk-iteration loop
    # --------------------------
    n_phases = number_of_phases
    sigmaij = species
    initial_density_guess = [np.ones(N) * 0.1 for _ in range(n_phases)]
    vij_per_phase = [compute_vij_for_mode(initial_density_guess[p], vk_mode_initial) for p in range(n_phases)]
    last_vij_concat = np.concatenate([v.reshape(-1) for v in vij_per_phase])

    final_solution = None
    global CURRENT_VIJ_PER_PHASE
    CURRENT_VIJ_PER_PHASE = vij_per_phase

    for outer_iter in range(1, max_outer_iters + 1):
        if verbose:
            print(f"[outer] iteration {outer_iter} - solving coexistence with current vij")
        CURRENT_VIJ_PER_PHASE = vij_per_phase
        sol = solve_general_coexistence(
            n_phases, sigmaij, mue_funcs, pressure_func,
            pvec,
            n_attempts=500000, verbose=verbose
        )
        if sol is None:
            raise RuntimeError("Coexistence solver failed to find a solution for current vij")
        rhos_per_phase = sol["rhos_per_phase"]
        
        print (rhos_per_phase)
     
        
        # compute density-based convergence metric
        if outer_iter > 1:
            drho_concat = np.concatenate([
                (np.array(rhos_per_phase[p]).reshape(-1) - np.array(last_rhos_per_phase[p]).reshape(-1))
                for p in range(n_phases)
            ])
            drho_norm = np.linalg.norm(drho_concat)
            if verbose:
                print(f"[outer] drho_norm = {drho_norm:.3e}")
            if drho_norm < tol_outer:
                if verbose:
                    print("[outer] Converged (drho_norm < tol_outer)")
                final_solution = sol
                break
        new_vij_per_phase = [ compute_vij_for_mode(np.array(rhos_per_phase[p]), vk_mode_iter)  for p in range(n_phases) ]
        if verbose:
            print(rhos_per_phase)
        vij_per_phase = new_vij_per_phase
        last_rhos_per_phase = [np.array(r).copy() for r in rhos_per_phase]
        final_solution = sol
    else:
        if verbose:
            print("[outer] Reached maximum outer iterations without full convergence")

    # --------------------------
    # 13) Save final solution (with μ/P evaluated)
    # --------------------------
    if final_solution is None:
        raise RuntimeError("No solution found")

    out = {
        "species": sigmaij,
        "ensemble": "isochore",
        "n_phases": n_phases,
        "rhos_per_phase": final_solution["rhos_per_phase"],
        "mu_per_phase": final_solution["mu_per_phase"],
        "pressure_per_phase": final_solution["pressure_per_phase"],
        "vij_per_phase": [v.tolist() for v in vij_per_phase],
        "vk_mode_initial": vk_mode_initial,
        "vk_mode_iter": vk_mode_iter
    }

    out_file = scratch / "Solution_coexistence_isochore.json"
    with open(out_file, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: np.asarray(x).tolist())

    if verbose:
        print(f"[isochore] solution written to: {out_file}")

    return out

