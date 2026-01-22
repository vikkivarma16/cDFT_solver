# cdft_solver/generators/rdf_isotropic.py

import json
import numpy as np
from scipy.fftpack import dst, idst
from scipy.interpolate import interp1d
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import re
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from collections.abc import Mapping
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.calculators.radial_distribution_function.closure import closure_update_c_matrix
from scipy.interpolate import interp1d
import os
import sys
import ctypes
from ctypes import c_double, c_int, POINTER


hard_core_repulsion = 1e4

# -------------------------------
# Locate shared library reliably
# -------------------------------
_here = os.path.dirname(__file__)

if sys.platform == "darwin":
    _libname = "liboz_radial.dylib"
elif sys.platform == "win32":
    _libname = "liboz_radial.dll"
else:
    _libname = "liboz_radial.so"

_lib_path = os.path.join(_here, _libname)

if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Shared library not found: {_lib_path}")

# Load shared library
lib = ctypes.CDLL(_lib_path)

# -----------------------------
# DST-based Hankel forward/inverse transforms
# -----------------------------
# cdft_solver/rdf/supplied_rdf.py


def solve_oz_kspace(h_k, densities, eps=1e-12):
    """
    Solve multi-component OZ equation in k-space:

        C(k) = H(k) [I + ρ H(k)]^{-1}

    Parameters
    ----------
    h_k : np.ndarray
        Total correlation in k-space, shape (N, N, Nk)
    densities : np.ndarray
        Species densities, shape (N,)
    eps : float
        Small regularization to stabilize inversion

    Returns
    -------
    c_k : np.ndarray
        Direct correlation in k-space, shape (N, N, Nk)
    """

    N, _, Nk = h_k.shape
    rho = np.diag(densities)

    c_k = np.zeros_like(h_k)

    I = np.eye(N)

    for ik in range(Nk):
        H = h_k[:, :, ik]

        # OZ matrix
        M = I + rho @ H

        # Regularized inverse
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            Minv = np.linalg.inv(M + eps * I)

        c_k[:, :, ik] = H @ Minv

    return c_k
    
# --------------------------------------------------
# Define function signature
# --------------------------------------------------

lib.solve_oz_matrix.argtypes = [
    c_int,                  # N
    c_int,                  # Nr
    POINTER(c_double),      # r
    POINTER(c_double),      # densities
    POINTER(c_double),      # c_r
    POINTER(c_double),      # gamma_r
]


def solve_oz_matrix(c_r, r, densities):
    N, _, Nr = c_r.shape
    gamma_r = np.zeros_like(c_r)

    # Flatten arrays for C
    c_r_flat = c_r.ravel()
    gamma_r_flat = gamma_r.ravel()
    
    #print("Flatten check:",
    #  c_r[1,1,3],
    #  c_r_flat[1*Nr*N + 1*Nr + 3])
    

    lib.solve_oz_matrix(
        c_int(N),
        c_int(Nr),
        r.ctypes.data_as(POINTER(c_double)),
        densities.ctypes.data_as(POINTER(c_double)),
        c_r_flat.ctypes.data_as(POINTER(c_double)),
        gamma_r_flat.ctypes.data_as(POINTER(c_double)),
    )

    # Reshape back to 3D
    gamma_r = gamma_r_flat.reshape((N, N, Nr))
    
    gamma_r = np.clip(gamma_r, -50.0, 50.0)

    return gamma_r


def multi_component_oz_solver_alpha(
    r,
    pair_closures,
    densities,
    u_matrix,
    sigma_matrix=None,
    c_update_flag=None,
    c_initial=None,
    n_iter=10000,
    tol=1e-8,
    alpha_rdf_max=0.1,
):
    """
    Multi-component OZ solver with adaptive alpha-mixing
    and selective c_ij update control.
    """

    if u_matrix is None:
        raise ValueError("u_matrix must be provided.")

    N = pair_closures.shape[0]
    Nr = len(r)

    # -----------------------------
    # Initialize arrays
    # -----------------------------
    gamma_r = np.zeros((N, N, Nr))

    if c_initial is not None:
        c_r = c_initial.copy()
    else:
        c_r = np.zeros((N, N, Nr))

    # Default: update all c_ij
    if c_update_flag is None:
        c_update_flag = np.ones((N, N), dtype=bool)
    else:
        c_update_flag = c_update_flag.astype(bool)

    print(f"\n🚀 Starting OZ solver (adaptive α, α_max = {alpha_rdf_max})")
    print(f"{'Iter':>6s} | {'Δγ(max)':>12s} | {'α':>6s}")

    prev_diff = np.inf
    alpha = min(0.01, alpha_rdf_max)

    # -----------------------------
    # Iteration loop
    # -----------------------------
    
    print ()
    for step in range(n_iter):

        # --- Closure update (ONLY where allowed)
        c_trial = closure_update_c_matrix(
            gamma_r, r, pair_closures, u_matrix, sigma_matrix
        )

        for i in range(N):
            for j in range(N):
                if c_update_flag[i, j]:
                    c_r[i, j, :] = c_trial[i, j, :]
                # else: keep frozen c_r[i,j,:]

        # --- Solve OZ
        gamma_new = solve_oz_matrix(c_r, r, densities)

        # --- Convergence check
        delta_gamma = gamma_new - gamma_r
        diff = np.max(np.abs(delta_gamma))

        # --- Adaptive mixing
        if diff < prev_diff:
            alpha = min(alpha * 1.05, alpha_rdf_max)
        else:
            alpha = max(alpha * 0.5, 1e-4)

        gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new

        if step % 100 == 0 or diff < tol:
            print(f"{step:6d} | {diff:12.3e} | {alpha:6.4f}")

        if diff < tol:
            print(f"\n✅ Converged in {step+1} iterations.")
            break

        prev_diff = diff

    if (diff>tol):
        print(f"\n⚠️ Warning: not converged after {n_iter} iterations.")

    # -----------------------------
    # Final observables
    # -----------------------------
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r


def wca_split(r, u):
    """
    Returns repulsive part of u(r) using WCA splitting.
    """
    idx_min = np.argmin(u)
    r_min = r[idx_min]
    u_min = u[idx_min]

    u_rep = np.zeros_like(u)
    mask = r <= r_min
    u_rep[mask] = u[mask] - u_min
    u_rep[~mask] = 0.0
    
    return u_rep


def detect_sigma_from_gr(r, g, g_tol=1e-3, min_width=3):
    """
    Detect hard-core diameter ONLY if g(r) is strictly zero
    over a finite r-interval.
    Returns sigma or None.
    """
    zero_mask = np.abs(g) < g_tol

    if not np.any(zero_mask):
        return 0.0

    # Find first non-zero region
    idx = np.where(~zero_mask)[0]
    if len(idx) == 0:
        return 0.0

    i0 = idx[0]

    # Require at least `min_width` zero points before σ
    if i0 < min_width:
        return 0.0

    return r[i0]


def boltzmann_potential_from_gr(g, beta=1.0, g_min=1e-8):
    """
    u(r) = -ln g(r) with numerical protection
    """
    g_safe = np.maximum(g, g_min)
    return -np.log(g_safe) / beta
    
    
    
    
def detect_first_minimum_near_core(r, u_ij, sigma=None):
    """
    Detect the first local minimum of u(r) after the hard core.
    If sigma is provided, search starts slightly above sigma.
    """

    # Exclude hard-core region
    if sigma is not None and sigma > 0:
        mask = r > 1.05 * sigma
    else:
        mask = r > r[1]

    r_use = r[mask]
    u_use = u_ij[mask]

    # Finite difference derivative
    du = np.gradient(u_use, r_use)

    # Find zero-crossings of derivative (− → +)
    for k in range(1, len(du) - 1):
        if du[k - 1] < 0 and du[k] > 0:
            return r_use[k], u_use[k]

    # Fallback: global minimum (safe)
    idx = np.argmin(u_use)
    return r_use[idx], u_use[idx]
    
    
    


def process_supplied_rdf_multistate(supplied_data, species, r_grid):

    if supplied_data is None:
        return {}

    # Locate state container (flexible naming)
    state_block = (
        supplied_data.get("states")
        or supplied_data.get("state")
        or supplied_data
    )

    states_out = {}
    N = len(species)
    Nr = len(r_grid)

    for state_name, state_data in state_block.items():

        # -----------------------------
        # Required metadata
        # -----------------------------
        densities_raw = state_data.get("densities", None)
        if densities_raw is None:
            raise KeyError(f"State '{state_name}' missing 'densities'")

        # ---------------------------------------
        # Case 1: dict keyed by species name
        # ---------------------------------------
        if isinstance(densities_raw, dict):
            densities = np.zeros(N, dtype=float)

            for i, s in enumerate(species):
                if s in densities_raw:
                    densities[i] = float(densities_raw[s])
                else:
                    densities[i] = 0.0   # default for missing species

        # ---------------------------------------
        # Case 2: scalar density → broadcast
        # ---------------------------------------
        elif np.isscalar(densities_raw):
            densities = np.full(N, float(densities_raw))

        # ---------------------------------------
        # Case 3: array-like
        # ---------------------------------------
        else:
            densities = np.asarray(densities_raw, dtype=float)

            if densities.ndim != 1:
                raise ValueError(
                    f"State '{state_name}' densities must be 1D array, scalar, or dict"
                )

            if len(densities) == 1:
                densities = np.full(N, densities[0])

            if len(densities) != N:
                raise ValueError(
                    f"State '{state_name}' densities size mismatch: "
                    f"expected {N}, got {len(densities)}"
                )


        # Beta / temperature
        if "beta" in state_data:
            beta = float(state_data["beta"])
        elif "temperature" in state_data:
            beta = 1.0 / float(state_data["temperature"])
        else:
            beta = 1.0  # default if nothing provided

        # -----------------------------
        # RDF dictionary
        # -----------------------------
        rdf_dict = state_data.get("rdf", {})
        if rdf_dict is None:
            raise KeyError(f"State '{state_name}' has no RDF data")

        # -----------------------------
        # Allocate arrays
        # -----------------------------
        g_target = np.zeros((N, N, Nr))
        fixed_mask = np.zeros((N, N), dtype=bool)

        # -----------------------------
        # Process all pairwise RDFs
        # -----------------------------
        for i, si in enumerate(species):
            for j, sj in enumerate(species):

                # pair keys may be "AB" or "BA"
                pair_keys = [f"{si}{sj}", f"{sj}{si}"]
                entry = None
                for key in pair_keys:
                    if key in rdf_dict:
                        entry = rdf_dict[key]
                        break
                if entry is None:
                    continue  # skip missing pairs

                r_sup = np.asarray(entry.get("x", entry.get("r", [])), dtype=float)
                g_sup = np.asarray(entry.get("y", entry.get("g", [])), dtype=float)

                if r_sup.size == 0 or g_sup.size == 0:
                    continue  # skip empty data

                # Interpolation
                interp = interp1d(
                    r_sup,
                    g_sup,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(g_sup[0], g_sup[-1]),
                )
                g_interp = interp(r_grid)

                # Symmetric assignment
                g_target[i, j, :] = g_target[j, i, :] = g_interp
                fixed_mask[i, j] = fixed_mask[j, i] = True

        states_out[state_name] = {
            "densities": densities,
            "beta": beta,
            "g_target": g_target,
            "fixed_mask": fixed_mask,
        }

    return states_out


def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None


def plot_u_matrix(r, u_matrix, species, outdir, filename="u_matrix.png"):
    """
    Plot and export u_ij(r) for all species pairs.

    Parameters
    ----------
    r : (Nr,) ndarray
        Radial grid (must match u_matrix)
    u_matrix : (N, N, Nr) ndarray
        Pair potential matrix
    species : list[str]
        Species labels, length N
    outdir : str or Path
        Output directory
    filename : str
        Output image filename
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    N = len(species)

    fig, axes = plt.subplots(
        N, N,
        figsize=(3.2 * N, 3.2 * N),
        sharex=True,
        sharey=False,
    )

    if N == 1:
        axes = [[axes]]

    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            ax = axes[i][j]

            u = u_matrix[i, j]

            ax.plot(r, u, lw=1.8)
            ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)

            ax.set_title(f"{si}–{sj}", fontsize=10)
            ax.grid(alpha=0.3)

            if i == N - 1:
                ax.set_xlabel("r")
            if j == 0:
                ax.set_ylabel(r"$u(r)$")

    fig.suptitle("Pair Potentials $u_{ij}(r)$", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = outdir / filename
    fig.savefig(outpath, dpi=600)
    plt.close(fig)

    print(f"✅ u(r) matrix plot saved to: {outpath}")







    
    
    
def boltzmann_inversion_standard(
    ctx,
    rdf_config,
    supplied_data,
    filename_prefix="multistate",
    export_plot=True,
    export_json=True
):
    """
    Multistate Isotropic Boltzmann / IBI inversion
    using OZ + closure.
    """

    g_floor = 1e-8
    hard_core_repulsion = 1e8

    # -----------------------------
    # Extract RDF parameters
    # -----------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    if rdf_block is None:
        raise KeyError("No 'rdf' key found in rdf_config")

    #params = rdf_config["rdf_parameters"]
    species = find_key_recursive(rdf_config, "species")
    N = len(species)

    beta_ref = rdf_block.get("beta", 1.0)
    beta = beta_ref
    tolerance = rdf_block.get("tolerance", 1e-6)
    ibi_tolerance = rdf_block.get("ibi_tolerance", 1e-6)                 
    n_iter = find_key_recursive(rdf_config, "max_iteration")
    alpha_max = rdf_block.get("alpha_max", 0.05)
    alpha_ibi_max = rdf_block.get("alpha_ibi_max", 0.05)
    
    n_iter_ibi = rdf_block.get("max_iteration_ibi", 500)
    
    system = rdf_config
    hc_data = hard_core_potentials(
        ctx=ctx,
        input_data=system,
        grid_points=5000,
        file_name_prefix="supplied_data_potential_hc.json",
        export_files=False
    )

    mf_data = meanfield_potentials(
        ctx=ctx,
        input_data=system,
        grid_points=5000,
        file_name_prefix="supplied_data_potential_mf.json",
        export_files=False
    )

    total_data = total_potentials(
        ctx=ctx,
        hc_source= hc_data,
        mf_source= mf_data,
        file_name_prefix="supplied_data_potential_total.json",
        export_files=False,
       
    )
    
    sigma = hc_data["sigma"]
    print ("sigma matrix before gr: ",sigma)
    

    # -----------------------------
    # Build r grid
    # -----------------------------
    
    n_points = rdf_block.get("n_points", 300)
    r_max  =  rdf_block.get("r_max", 6)
    dr = r_max / (n_points + 1)
    r = dr * np.arange(1, n_points + 1)
    
    # -----------------------------
    # Closures (all ON initially)
    # -----------------------------
    closure_cfg = rdf_block.get("closure", {})
    pair_closures = np.empty((N, N), dtype=object)

    n = len(species)
    pair_closures = np.empty((n, n), dtype=object)

    for i, si in enumerate(species):
        for j in range(i, n):   # <-- j starts from i
            sj = species[j]

            key_ij = f"{si}{sj}"
            key_ji = f"{sj}{si}"

            if key_ij in closure_cfg:
                closure = closure_cfg[key_ij]
            elif key_ji in closure_cfg:
                closure = closure_cfg[key_ji]
            else:
                raise KeyError(
                    f"Missing closure for pair '{key_ij}' or '{key_ji}'"
                )

            # assign symmetrically
            pair_closures[i, j] = closure
            pair_closures[j, i] = closure


    # -----------------------------
    # Potentials
    # -----------------------------
    
    potential_dict = total_data["total_potentials"]
    u_matrix = np.zeros((N, N, len(r)))
    print ("closures applied: ", pair_closures)
    N = len (species)
    n = len (species)
    
    
    states = process_supplied_rdf_multistate(
        supplied_data, species, r
    )
    if not states:
        raise ValueError("No multistate RDF data provided")

    state_names = list(states.keys())
    beta_ref = states[state_names[0]]["beta"]
    
    
    print ("reference beta: ", beta_ref)

    # Uniform state weights (can be changed later)
    w_state = {s: 1.0 / len(states) for s in states}

    # -------------------------------------------------
    # Initialize σ and u
    # -------------------------------------------------
    sigma_matrix = np.zeros((N, N)) if sigma is None else sigma.copy()
    invert_mask = np.ones((N, N), dtype=bool)

    for i, si in enumerate(species):
        for j in range(i, N):   # fixed N
            sj = species[j]

            key_ij = si + sj
            key_ji = sj + si

            pdata = (
                potential_dict.get(key_ij)
                or potential_dict.get(key_ji)
            )

            if pdata is None:
                continue

            # interpolate once
            interp_u = interp1d(
                pdata["r"],
                pdata["U"],
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )

            u_val = beta_ref * interp_u(r)

            # Correct non-zero detection
            if np.any(np.abs(u_val) > 1e-6):
                invert_mask[i, j] = invert_mask[j, i] = False

            # symmetric assignment
            u_matrix[i, j, :] = u_val
            u_matrix[j, i, :] = u_val

                
            
    plots = Path(ctx.plots_dir)      
    # u_matrix: (N, N, Nr), r: (Nr,)
    u_strength = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            u = u_matrix[i, j, :]
            integrand = r**2 * u
            u_strength[i, j] = 4.0 * np.pi * np.trapz(integrand, r)

    #print("Integrated potential strength (trapezoidal) for each pair:")
    #print(u_strength)


    plot_u_matrix( r=r, u_matrix=u_matrix, species=species, outdir=plots, filename="pair_potentials_before inversion.png",)
    # -----------------------------
    # Sigma matrix
    # -----------------------------
    sigma_matrix = np.zeros((N, N)) if sigma is None else np.array (sigma)

    # Initialize from first state RDF
    s0 = state_names[0]
    g0 = states[s0]["g_target"]
    
    enable_sigma_refinement = 0 
    initialized = np.zeros((N, N), dtype=bool)
    for sname, sdata in states.items():
        mask = sdata["fixed_mask"]
        for i in range(N):
            for j in range(N):
                if (mask[i, j] == True ):
                    g_safe = np.maximum(g0[i, j], g_floor)
                    u_matrix[i, j] = u_matrix[j, i] = -np.log(g_safe)
                    sigma_matrix[i, j] = sigma_matrix[j, i] = detect_sigma_from_gr(r, g0[i, j])
                    invert_mask[i, j] = invert_mask[j, i] = True
                    if (sigma_matrix[i, j] > 0.0):
                        enable_sigma_refinement =  1

    sigma_update_every = 1000000
    sigma_freeze_after = n_iter_ibi
    
    print ("sigma matrix detected before ibi: ",sigma_matrix)
    
    plot_u_matrix( r=r, u_matrix=u_matrix, species=species, outdir=plots, filename="pair_potentials_before_ibi.png",)
    
    
    # -------------------------------------------------
    # Adaptive IBI parameters
    # -------------------------------------------------
    alpha_ibi = 0.01

    alpha_min_ibi = 1e-4
    alpha_max_ibi = alpha_ibi_max   # keep your existing max
    alpha_power = 0.5               # controls adaptation strength

    max_diff_prev = np.inf
    sigma_ref  =  sigma_matrix.copy()
    # -------------------------------------------------
    # Multistate IBI loop
    # -------------------------------------------------
    
    storage_flag = False

    # -------------------------------------------------
    # Storage for final OZ results (used later for sigma)
    # -------------------------------------------------
    final_oz_results = {}
    for it in range(1, n_iter_ibi + 1):

        delta_u_accum = np.zeros_like(u_matrix)
        max_diff = 0.0

        # -----------------------------
        # State loop
        # -----------------------------
        for sname, sdata in states.items():

            beta_s = sdata["beta"]
            densities_s = sdata["densities"]
            g_target = sdata["g_target"]
            fixed_mask = sdata["fixed_mask"]

            print("\ntemperature in state", sname, ":", beta_s)

            c_r, gamma_r, g_pred = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities_s, float),
                u_matrix=beta_s * u_matrix / beta_ref,
                sigma_matrix=sigma_matrix,
                n_iter=n_iter,
                tol=tolerance,
                alpha_rdf_max=alpha_max,
            )

            g_pred_safe = np.maximum(g_pred, g_floor)
            g_target_safe = np.maximum(g_target, g_floor)

            for i in range(N):
                for j in range(N):

                    if not fixed_mask[i, j]:
                        continue

                    mask_r = g_target_safe[i, j] > 1e-4
                    delta_s = np.zeros_like(r)

                    delta_s[mask_r] = (beta_ref / beta_s) * np.log(
                        g_pred_safe[i, j, mask_r] /
                        g_target_safe[i, j, mask_r]
                    )

                    delta_u_accum[i, j] += w_state[sname] * delta_s

                    max_diff = max(
                        max_diff,
                        np.max(np.abs(g_pred[i, j] - g_target[i, j]))
                    )

        # -------------------------------------------------
        # Adaptive alpha IBI update
        # -------------------------------------------------
        if it % 10 == 0 and it > 1 and max_diff > 0.0:
            ratio = max_diff_prev / max_diff
            ratio = np.clip(ratio, 0.2, 5.0)

            alpha_ibi *= ratio ** alpha_power
            alpha_ibi = np.clip(alpha_ibi, alpha_min_ibi, alpha_max_ibi)

        max_diff_prev = max_diff

        # -------------------------------------------------
        # Apply combined potential update
        # -------------------------------------------------
        for i in range(N):
            for j in range(N):
                if invert_mask[i, j]:
                    u_matrix[i, j] += alpha_ibi * delta_u_accum[i, j]

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------
        print(
            f"IBI iter {it:6d} | max|Δg| = {max_diff:12.3e} | α = {alpha_ibi:7.4f}"
        )

        # -------------------------------------------------
        # Convergence check
        # -------------------------------------------------
        if max_diff < ibi_tolerance:
            print(f"\n✅ Multistate IBI converged in {it} iterations.")
            storage_flag = True

        # -------------------------------------------------
        # FINAL STORAGE PASS (one extra OZ solve)
        # -------------------------------------------------
        
        if storage_flag:
            print("\n📦 Storing final OZ results for sigma analysis...")

            final_oz_results.clear()
            c_pred =  np.zeros_like (g_pred) 
            gamma_pred =  np.zeros_like (g_pred)

            for sname, sdata in states.items():
                beta_s = sdata["beta"]
                densities_s = sdata["densities"]
                
                c_pred, gamma_pred, g_pred = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(densities_s, float),
                    u_matrix=beta_s * u_matrix / beta_ref,
                    sigma_matrix=sigma_matrix,
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )

                final_oz_results[sname] = {
                    "beta": beta_s,
                    "densities": np.asarray(densities_s, float),
                    "g_pred": g_pred.copy(),
                    "c_pred": c_pred.copy(),
                    "gamma_pred": gamma_pred.copy(),
                }

            break

    else:
        print("\n⚠️ Multistate IBI did not converge.")
        
    # -------------------------------------------------
    # Export
    # -------------------------------------------------
    
    # -------------------------------------------------
    # PHASE A: Detect hard-core pairs from g(r)
    # -------------------------------------------------

    

    
    
    
    
    
    #import numpy as np

    def compute_bh_radius_shift_truncated(r, u_r, beta):
        """
        Compute Barker–Henderson (BH) hard-core diameter
        with integration truncated at the first zero of the
        WCA reference potential.

        Parameters
        ----------
        r : (nr,) ndarray
            Radial grid (assumed increasing, starting near 0).
        u_r : (nr,) ndarray
            Pair potential u(r).
        beta : float
            Inverse temperature (1 / kT).

        Returns
        -------
        d_bh : float
            Barker–Henderson diameter.
        r0 : float
            First zero-crossing of the WCA reference potential.
        """

        # WCA reference (purely repulsive part)
        u_ref = wca_split(r, u_r)   # must return repulsive reference only

        # Locate first zero crossing (repulsive → attractive)
        zero_mask = u_ref <= 0.0
        if not np.any(zero_mask):
            raise RuntimeError("WCA reference potential never crosses zero.")

        idx0 = np.argmax(zero_mask)   # first True index
        r0 = r[idx0]

        # Truncate integration domain
        r_trunc = r[:idx0 + 1]
        u_trunc = u_ref[:idx0 + 1]

        # Barker–Henderson integrand
        integrand = 1.0 - np.exp(-beta * u_trunc)

        # BH diameter
        d_bh = np.trapz(integrand, r_trunc)

        return d_bh, r0

        
        
        
    def compute_bh_radius_truncated(r, u_r, beta):
        """
        Barker–Henderson radius with integration truncated
        at the first zero of u(r).
        """
        idx_zero = np.where(u_r <= 0.001)[0]
        
        r_tab = r

        if len(idx_zero) > 0:
            # Take the first r where potential becomes zero
            r_max = r_tab[idx_zero[0]]
        else:
            # fallback if potential never crosses zero
            r_max = r_tab.max()
        r0 =  r_max
        mask = r <= r_max
        integrand = 1.0 - np.exp(-beta * u_r[mask])

        return np.trapz(integrand, r[mask]), r0
        
    

    sigma_guess = np.zeros((N, N))
    has_core = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i, N):

            sigma_candidates = []

            for sname, res in final_oz_results.items():
                g_ij = res["g_pred"][i, j]
                sigma_s = detect_sigma_from_gr(r, g_ij)
                if sigma_s > 0.0:
                    sigma_candidates.append(sigma_s)

            if sigma_candidates:
                sigma_ij = max(sigma_candidates)  # conservative
                sigma_guess[i, j] = sigma_guess[j, i] = sigma_ij
                has_core[i, j] = has_core[j, i] = True

                print(f"Detected hard core for pair ({i},{j}) : σ ≈ {sigma_ij:.4f}")


    hard_core_pairs = [(i, j) for i in range(N) for j in range(i, N) if has_core[i, j]]

    def hard_core_potential(r, sigma, U0=1e6):
                u = np.zeros_like(r)
                u[r < sigma] = U0
                return u

    def build_hard_core_u_from_sigma(sigma_mat):
        u = np.zeros_like(u_matrix)
        for i in range(N):
            for j in range(i, N):
                if has_core[i, j]:
                    u[i, j] = hard_core_potential(r, sigma_mat[i, j])
                else:
                    u[i, j] = u_matrix[i, j].copy()
                u[j, i] =  u[i, j]
        return u

    total_pair = [ (i, j) for i in range(N) for j in range(i, N) ]
    
    
    
    
    
    
    
    
    
    
    
    
    
    if hard_core_pairs:

        print("\n🔧 Starting sigma calibration stage...")
        # -------------------------------------------------
        # PHASE B: Build WCA-repulsive reference potential, which could be computed on the same basis where it was derived to be the same number of particles... now the computatoin for
        # -------------------------------------------------

        u_ref = np.zeros_like(u_matrix)   
        for i in range(N):
            for j in range(N):
                if has_core[i, j]:
                    u_ref[i, j] = wca_split(r, u_matrix[i, j])
                    # u_r =  u_matrix[i, j]
                    # bh , r0 =  compute_bh_radius_truncated(r, u_r, beta_ref)
                    # mask  =  r < r0
                    # u_ref[i ,j] =  np.zeros_like(r)
                    # u_ref[i, j, mask] =  u_matrix[i, j, mask]
                else:
                    u_ref[i, j] = u_matrix[i, j].copy()
        
        
        # -------------------------------------------------
        # PHASE C: Compute reference RDFs for ALL states
        # -------------------------------------------------

        g_ref = {}
        c_ref = {}
        gamma_ref = {}
        for sname, sdata in states.items():

            beta_s = sdata["beta"]
            rho_s = sdata["densities"]
            print(f"\nComputing reference RDF for state {sname}")
            c_ref_state, gamma_ref_state, g_ref_state = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(rho_s, float),
                u_matrix=beta_s * u_ref / beta_ref,
                sigma_matrix=np.zeros((N, N)),
                n_iter=n_iter,
                tol=tolerance,
                alpha_rdf_max=alpha_max,
            )

            g_ref[sname] = g_ref_state
            c_ref[sname] = c_ref_state
            gamma_ref[sname] = gamma_ref_state  

        # -------------------------------------------------
        # PHASE D: Collective sigma optimization
        # -------------------------------------------------

        def unpack_sigma_vector(sigma_vec):
            sigma_mat = np.zeros((N, N))
            k = 0
            for (i, j) in hard_core_pairs:
                sigma_mat[i, j] = sigma_mat[j, i] = sigma_vec[k]
                k += 1
            return sigma_mat

        

        def sigma_objective(sigma_vec):
            sigma_mat = unpack_sigma_vector(sigma_vec)
            u_trial = build_hard_core_u_from_sigma(sigma_mat)
            loss = 0.0

            for sname, sdata in states.items():
                beta_s = sdata["beta"]
                rho_s = sdata["densities"]
                _, _, g_trial = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(rho_s, float),
                    u_matrix=beta_s * u_trial / beta_ref,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )

                for (i, j) in total_pair:
                    diff = g_trial[i, j] - g_ref[sname][i, j]
                    loss += np.sum(diff * diff)

            return loss

        # -------------------------------------------------
        # Run optimizer
        # -------------------------------------------------

        sigma_init_vec = np.array([sigma_guess[i, j] for (i, j) in hard_core_pairs])
        print("\nOptimizing sigma collectively across all states and pairs...")
        result = minimize(
            sigma_objective,
            sigma_init_vec,
            method="Powell",
            options={"xtol": 1e-6, "ftol": 1e-6, "disp": True},
        )
        sigma_opt = unpack_sigma_vector(result.x)
        
        
        
        
        
        bh_zero = {}
        bh_sigma  =  np.zeros_like(sigma_opt)
        for (i, j) in hard_core_pairs:
            d_bh, r0 = compute_bh_radius_truncated(
                r,
                u_matrix[i, j],   # or u_repulsive_wca[i,j]
                beta_ref
            )
            bh_zero[(i, j)] = r0
            bh_sigma[i, j] = d_bh
            bh_sigma[j, i] =  bh_sigma[i, j]
            
        
        
        def compute_repulsive_gr(sigma_mat):
            """
            Compute RDFs using ONLY hard-core repulsion defined by sigma_mat.
            """
            g_store = {}
            c_store = {}
            gamma_store = {}
            u_rep = build_hard_core_u_from_sigma(sigma_mat)

            for sname, sdata in states.items():
                c_state, gamma_state, g_state = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(sdata["densities"], float),
                    u_matrix=sdata["beta"] * u_rep / beta_ref,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )
                g_store[sname] = g_state
                c_store[sname] = c_state
                gamma_store[sname] = gamma_state

            return g_store, c_store, gamma_store


        # Repulsive RDFs using optimized sigma
        g_rep_sigma_opt, c_rep_sigma_opt, gamma_rep_sigma_opt = compute_repulsive_gr(sigma_opt)
        # Repulsive RDFs using Barker–Henderson sigma
        g_rep_sigma_bh, c_rep_sigma_bh, gamma_rep_sigma_bh = compute_repulsive_gr(bh_sigma)
        # ============================================================
        # PHASE 7 — Save reference package (JSON)
        # ============================================================
        reference_package = {
            # --- sigmas ---
            "sigma_opt": sigma_opt.tolist(),
            "sigma_bh": bh_sigma.tolist(),

            # --- BH metadata ---
            "bh_meta": {
                f"{i},{j}": {
                    "d_bh": bh_sigma[i, j],
                    "r0": bh_zero[(i, j)],
                }
                for (i, j) in hard_core_pairs
            },

            # --- reference RDFs ---
            "r" : r.tolist(), 
            "g_ref_hard": {k: v.tolist() for k, v in g_ref.items()},
            "c_ref_hard": {k: v.tolist() for k, v in c_ref.items()},
            "gamma_ref_hard": {k: v.tolist() for k, v in gamma_ref.items()},
            
            "g_rep_sigma_opt": {k: v.tolist() for k, v in g_rep_sigma_opt.items()},
            "c_rep_sigma_opt": {k: v.tolist() for k, v in c_rep_sigma_opt.items()},
            "gamma_rep_sigma_opt": {k: v.tolist() for k, v in gamma_rep_sigma_opt.items()},
            
            "g_rep_sigma_bh": {k: v.tolist() for k, v in g_rep_sigma_bh.items()},
            "c_rep_sigma_bh": {k: v.tolist() for k, v in c_rep_sigma_bh.items()},
            "gamma_rep_sigma_bh": {k: v.tolist() for k, v in gamma_rep_sigma_bh.items()},

            # --- target RDFs ---
            "g_real": {
                k: v["g_pred"].tolist() for k, v in final_oz_results.items()
            },
            
            "c_real": {
                k: v["c_pred"].tolist() for k, v in final_oz_results.items()
            },
            
             "gamma_real": {
                k: v["gamma_pred"].tolist() for k, v in final_oz_results.items()
            },
        }


        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_file = out / "result_sigma_analysis.json"
        with open(json_file, "w") as f:
            json.dump(reference_package, f, indent=4)

        print("✅ Saved Sigma_package.json")

                
        
        
        
        
        # ============================================================
        # PHASE 8 — WCA splitting RDFs (σ_BH vs σ_opt)
        # ============================================================

        def compute_wca_gr(sigma_mat):
            """
            Compute RDFs after WCA splitting using a given hard-core sigma.
            Repulsive part: hard core from sigma_mat
            Attractive part: shifted tail beyond first minimum
            """
            u_repulsive = build_hard_core_u_from_sigma(sigma_mat)
            u_attractive = np.zeros_like(u_matrix)
            u_wca_total = np.zeros_like(u_matrix)
            r_minima = {}
            attractive_pairs = [ (i, j) for i in range(N) for j in range(i, N) if has_core[i, j] and np.any(u_matrix[i, j] < -1e-4) ]
            for i in range(N):
                for j in range(i, N):
                    if (i, j) in attractive_pairs:
                        # Detect first minimum close to the hard core
                        r_m, u_m = detect_first_minimum_near_core(
                            r,
                            u_matrix[i, j],
                            sigma=sigma_mat[i, j],
                        )
                        r_minima[(i, j)] = r_m
                        u_att = np.zeros_like(r)
                        mask_rep = r <= r_m
                        mask_att = r > r_m
                        # WCA attractive tail
                        u_att[mask_rep] = u_m
                        u_att[mask_att] = u_matrix[i, j][mask_att]
                        u_attractive[i, j] = u_att
                        u_attractive[j, i] = u_att
                        u_wca_total[i, j] = u_repulsive[i, j] + u_att
                        u_wca_total[j, i] = u_wca_total[i, j]
                    else:
                        # No hard core → keep original interaction
                        u_wca_total[i, j] = u_matrix[i, j]
                        u_wca_total[j, i] = u_matrix[j, i]

            # ---- Compute RDFs for all states ----
            g_wca = {}
            c_wca = {}
            gamma_wca = {}
            for sname, sdata in states.items():
                c_state, gamma_state, g_state = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(sdata["densities"], float),
                    u_matrix=sdata["beta"] * u_wca_total / beta_ref,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )
                g_wca[sname] = g_state
                c_wca[sname] = c_state
                gamma_wca[sname] = gamma_state

            return g_wca, c_wca, gamma_wca, r_minima, u_repulsive, u_attractive


        # ------------------------------------------------------------
        # Compute WCA RDFs for both sigma definitions
        # ------------------------------------------------------------

        g_wca_sigma_bh, c_wca_sigma_bh, gamma_wca_sigma_bh, rmin_bh, u_repulsive_bh, u_attractive_bh  = compute_wca_gr(bh_sigma)
        g_wca_sigma_opt, c_wca_sigma_opt, gamma_wca_sigma_opt, rmin_opt, u_repulsive_opt, u_attractive_opt = compute_wca_gr(sigma_opt)


        # ============================================================
        # PHASE 9 — Export WCA diagnostics to JSON
        # ============================================================

        wca_package = {
            "sigma_bh": bh_sigma.tolist(),
            "sigma_opt": sigma_opt.tolist(),
            
            "r" : r.tolist(),

            "rmin_bh": {f"{i},{j}": float(v) for (i, j), v in rmin_bh.items()},
            "rmin_opt": {f"{i},{j}": float(v) for (i, j), v in rmin_opt.items()},
            "u_attractive_bh" : u_attractive_bh.tolist(),
            "u_repulsive_bh" : u_repulsive_bh.tolist(),
            "u_attractive_opt" : u_attractive_opt.tolist(),
            "u_repulsive_opt" : u_repulsive_opt.tolist(),
            "u_total" : u_matrix.tolist(),
            "g_wca_sigma_bh": {k: v.tolist() for k, v in g_wca_sigma_bh.items()},
            "c_wca_sigma_bh": {k: v.tolist() for k, v in c_wca_sigma_bh.items()},
            "gamma_wca_sigma_bh": {k: v.tolist() for k, v in gamma_wca_sigma_bh.items()},
            
            
            "g_wca_sigma_opt": {k: v.tolist() for k, v in g_wca_sigma_opt.items()},
            "c_wca_sigma_opt": {k: v.tolist() for k, v in c_wca_sigma_opt.items()},
            "gamma_wca_sigma_opt": {k: v.tolist() for k, v in gamma_wca_sigma_opt.items()},
            
            "g_real": {
                k: v["g_pred"].tolist() for k, v in final_oz_results.items()
            },
            
            "c_real": {
                k: v["c_pred"].tolist() for k, v in final_oz_results.items()
            },
            
             "gamma_real": {
                k: v["gamma_pred"].tolist() for k, v in final_oz_results.items()
            },
        }


        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_file = out / "result_wca_gr_comparison.json"

        with open(json_file, "w") as f:
            json.dump(wca_package, f, indent=4)

        print("✅ Saved wca_gr_comparison_package.json")

                
        
        
        
        
        
        
        
        
        # ============================================================
        # PHASE 10 — Attractive calibration (σ_opt vs σ_BH)
        # ============================================================

        def run_attractive_calibration(sigma_mat, label):
            """
            Run attractive-part calibration for a given sigma definition.
            Returns final attractive potential, total potential, and RDFs.
            """

            print(f"\n🔧 Starting attractive calibration using {label}")

            # -------------------------------------------------
            # Select sigma-fixed attractive pairs
            # -------------------------------------------------
            attractive_pairs = [
                (i, j)
                for i in range(N)
                for j in range(i, N)
                if has_core[i, j] and np.any(u_matrix[i, j] < -1e-4)
            ]

            if not attractive_pairs:
                print("No hard-core attractive pairs found.")
                return None

            # -------------------------------------------------
            # Repulsive hard-core part
            # -------------------------------------------------
            u_repulsive = build_hard_core_u_from_sigma(sigma_mat)

            # -------------------------------------------------
            # Initial attractive estimate (linearized OZ)
            # -------------------------------------------------
            u_attractive = np.zeros_like(u_matrix, dtype=float)

            eps = 1e-12
            num_state = 0

            for sname, sdata in states.items():
                beta_s = float(sdata["beta"])
                g_ref_state  = g_ref[sname]
                g_pred_state = final_oz_results[sname]["g_pred"]

                for (i, j) in attractive_pairs:
                    mask_r = r > sigma_mat[i, j]
                    safe_g_ref = np.maximum(g_ref_state[i, j], eps)

                    delta_u = (
                        beta_ref
                        * (g_ref_state[i, j] - g_pred_state[i, j])
                        / (safe_g_ref * beta_s)
                    )

                    u_attractive[i, j, mask_r] = delta_u[mask_r]
                    u_attractive[j, i, mask_r] = u_attractive[i, j, mask_r]

                num_state += 1

            u_attractive /= num_state

            # -------------------------------------------------
            # IBI refinement of attractive tail
            # -------------------------------------------------
            alpha_attr = 0.1
            u_attr_trial = u_attractive.copy()

            for it in range(1, n_iter_ibi):

                max_diff = 0.0
                delta_u_accum = np.zeros_like(u_attr_trial)

                for sname, sdata in states.items():
                    beta_s = sdata["beta"]
                    rho_s  = sdata["densities"]

                    # Build total potential
                    u_total = u_matrix.copy()

                    for (i, j) in hard_core_pairs:
                        u_total[i, j] = u_repulsive[i, j]
                        u_total[j, i] = u_total[i, j]

                    for (i, j) in attractive_pairs:
                        u_total[i, j] += u_attr_trial[i, j]
                        u_total[j, i] = u_total[i, j]

                    # Compute RDF
                    _, _, g_trial = multi_component_oz_solver_alpha(
                        r=r,
                        pair_closures=pair_closures,
                        densities=np.asarray(rho_s, float),
                        u_matrix=beta_s * u_total / beta_ref,
                        sigma_matrix=np.zeros((N, N)),
                        n_iter=n_iter,
                        tol=tolerance,
                        alpha_rdf_max=alpha_max,
                    )

                    # Accumulate corrections
                    for (i, j) in attractive_pairs:
                    
                        r_m, u_m = detect_first_minimum_near_core( r, u_matrix[i, j], sigma=sigma_mat[i, j])
                        mask_r_super  = r > r_m 
                        mask_r = r > sigma_mat[i, j]
                        delta = np.zeros_like(r)

                        delta[mask_r_super] = np.log(
                            g_trial[i, j, mask_r_super]
                            / final_oz_results[sname]["g_pred"][i, j, mask_r_super]
                        )

                        delta_u_accum[i, j] += delta
                        delta_u_accum[j, i] = delta_u_accum[i, j]

                        max_diff = max(
                            max_diff,
                            np.max(
                                np.abs(
                                    g_trial[i, j, mask_r_super]
                                    - final_oz_results[sname]["g_pred"][i, j, mask_r_super]
                                )
                            ),
                        )
                    
                    

                # Apply update + enforce WCA continuity
                for (i, j) in attractive_pairs:
                    u_attr_trial[i, j] += alpha_attr * delta_u_accum[i, j]

                    r_m, u_m = detect_first_minimum_near_core(
                        r, u_matrix[i, j], sigma=sigma_mat[i, j]
                    )

                    u_att = np.zeros_like(r)
                    u_att[r <= r_m] = u_m
                    u_att[r >  r_m] = u_attr_trial[i, j][r > r_m]

                    u_attr_trial[i, j] = u_att
                    u_attr_trial[j, i] = u_att

                print(f"[{label}] Attractive IBI iter {it:3d} | max|Δg| = {max_diff:10.3e}")

                if max_diff < ibi_tolerance:
                    print(f"✅ Attractive IBI converged for {label}")
                    break

            # -------------------------------------------------
            # Final RDFs with calibrated potential
            # -------------------------------------------------
            u_final = u_repulsive + u_attr_trial
            g_ur = {}
            c_ur = {}
            gamma_ur = {}

            for sname, sdata in states.items():
            
                c_state, gamma_state, g_state = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(sdata["densities"], float),
                    u_matrix=sdata["beta"] * u_final / beta_ref,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )
                g_ur[sname] = g_state
                c_ur[sname] = c_state
                gamma_ur[sname] =  gamma_state

            return {
                "u_attractive": u_attr_trial,
                "u_repulsive": u_repulsive
                "u_total": u_final,
                "g_ur": g_ur,
                "c_ur": c_ur,
                "gamma_ur": gamma_ur,
            }


        # ============================================================
        # Run for both sigma definitions
        # ============================================================

        results_sigma_opt = run_attractive_calibration(sigma_opt, "sigma_opt")
        results_sigma_bh  = run_attractive_calibration(bh_sigma,  "sigma_bh")


        # ============================================================
        # Export attractive-calibration package
        # ============================================================

        attractive_package = {
            "sigma_opt": sigma_opt.tolist(),
            "sigma_bh": bh_sigma.tolist(),
            
            "r" : r.tolist(), 

            "g_real": {
                k: v["g_pred"].tolist() for k, v in final_oz_results.items()
            },
            
            "c_real": {
                k: v["c_pred"].tolist() for k, v in final_oz_results.items()
            },
            
            "gamma_real": {
                k: v["gamma_pred"].tolist() for k, v in final_oz_results.items()
            },

            "sigma_opt_results": {
                "g_ur": {k: v.tolist() for k, v in results_sigma_opt["g_ur"].items()},
                "c_ur": {k: v.tolist() for k, v in results_sigma_opt["c_ur"].items()},
                "gamma_ur": {k: v.tolist() for k, v in results_sigma_opt["gamma_ur"].items()},
                "u_attractive": results_sigma_opt["u_attractive"].tolist(),
                "u_splitted": results_sigma_opt["u_total"].tolist(),
                "u_total": u_matrix.tolist(),
            },

            "sigma_bh_results": {
                "g_ur": {k: v.tolist() for k, v in results_sigma_bh["g_ur"].items()},
                "c_ur": {k: v.tolist() for k, v in results_sigma_bh["c_ur"].items()},
                "gamma_ur": {k: v.tolist() for k, v in results_sigma_bh["gamma_ur"].items()},
                "u_attractive": results_sigma_bh["u_attractive"].tolist(),
                "u_splitted": results_sigma_bh["u_total"].tolist(),
                "u_total": u_matrix.tolist(),
            },
        }

        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_file = out / "result_attractive_calibration.json"

        with open(json_file, "w") as f:
            json.dump(attractive_package, f, indent=4)

        print("✅ Saved attractive_calibration_comparison.json")
           
                
                        
    
    print ("Integrated strength kernel analysis is going on !!!!!!!")
    
    
    
    
    # ============================================================
    # Compute G(r) via λ-integration for each pair and state
    # ============================================================

    def compute_G_of_r(u_repulsive, u_attractive, states, r, pair_closures, beta_ref, N, n_alpha=20):
        """
        Computes G(r) = ∫_0^1 g_alpha(r) dα
        where u_alpha = u_repulsive + α * u_attractive
        """

        alpha_grid = np.linspace(0.0, 1.0, n_alpha)
        dalpha = alpha_grid[1] - alpha_grid[0]

        G_r_dict = {sname: np.zeros_like(u_attractive) for sname in states.keys()}

        # Loop over states
        for sname, sdata in states.items():
            beta_s = sdata["beta"]
            rho_s = np.asarray(sdata["densities"], float)

            G_accum = np.zeros_like(u_attractive)

            # Loop over λ (α)
            for alpha in alpha_grid:
                # Total potential for this α
                u_alpha = u_repulsive + alpha * u_attractive

                # Solve OZ for this u_alpha
                _, _, g_alpha = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=rho_s,
                    u_matrix=beta_s * u_alpha / beta_ref,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )

                # Accumulate for integration
                for i in range(N):
                    for j in range(N):
                        G_accum[i, j] += g_alpha[i, j] * dalpha

            # Store G(r) for this state
            G_r_dict[sname] = G_accum

        return G_r_dict

    # ============================================================
    # Run G(r) computation for σ_opt
    # ============================================================
    G_r_sigma_opt = compute_G_of_r(
        u_repulsive=results_sigma_opt ["u_repulsive"],
        u_attractive=results_sigma_opt["u_attractive"],
        states=states,
        r=r,
        pair_closures=pair_closures,
        beta_ref=beta_ref,
        N=N,
        n_alpha=20
    )

    # ============================================================
    # Run G(r) computation for σ_BH
    # ============================================================
    G_r_sigma_bh = compute_G_of_r(
        u_repulsive=results_sigma_bh["u_repulsive"],
        u_attractive=results_sigma_bh["u_attractive"],
        states=states,
        r=r,
        pair_closures=pair_closures,
        beta_ref=beta_ref,
        N=N,
        n_alpha=20
    )

    # ============================================================
    # Export results along with u_attractive
    # ============================================================
    attractive_package = {
        "sigma_opt": sigma_opt.tolist(),
        "sigma_bh": bh_sigma.tolist(),
        "r": r.tolist(),
        
        "G_r_sigma_opt": {k: v.tolist() for k, v in G_r_sigma_opt.items()},
        "G_r_sigma_bh":  {k: v.tolist() for k, v in G_r_sigma_bh.items()},
        
        "u_attractive_sigma_opt": results_sigma_opt["u_attractive"].tolist(),
        "u_attractive_sigma_bh": results_sigma_bh["u_attractive"].tolist()
    }

    out = Path(ctx.scratch_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_file = out / "result_G_of_r.json"

    with open(json_file, "w") as f:
        json.dump(attractive_package, f, indent=4)

    print("✅ G(r) and u_attractive exported →", json_file)

    
    
    
    
    
    
    
        
    if export_json:
        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        data = {"pairs": {}}
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                data["pairs"][f"{si}{sj}"] = {
                    "r": r.tolist(),
                    "u_r": (u_matrix[i, j] / beta_ref).tolist(),
                    "sigma": float(sigma_matrix[i, j]),
                }

        json_file = out / f"{filename_prefix}_potential.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Multistate inverted potential exported: {json_file}")

    # Plot export
    if export_plot:
        plots_dir = getattr(ctx, "plots_dir", ctx.scratch_dir)
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # 1. Potential plots
        # -------------------------------------------------
        for i, si in enumerate(species):
            for j, sj in enumerate(species):

                plt.figure(figsize=(6, 4))
                plt.plot(r, u_matrix[i, j] / beta_ref, label=f"{si}{sj}")

                if sigma_matrix[i, j] > 0:
                    plt.axvline(
                        sigma_matrix[i, j],
                        color="r",
                        linestyle="--",
                        label="σ"
                    )

                plt.xlabel("r")
                plt.ylabel("u(r)")
                plt.title(f"Inverted Potential: {si}{sj}")
                plt.legend()
                plt.tight_layout()
                plt.ylim (-10, 10)
                plt.savefig(plots_dir / f"{filename_prefix}_potential_{si}{sj}.png", dpi  = 600)
                plt.close()

        # -------------------------------------------------
        # 2. RDF comparison plots (target vs predicted)
        # -------------------------------------------------
        for sname, sdata in states.items():

            g_target = sdata["g_target"]
            fixed_mask = sdata["fixed_mask"]

            # Use last computed g_pred from OZ
            # (assumed to be from final iteration)
            g_pred_safe = np.maximum(g_pred, g_floor)
            g_target_safe = np.maximum(g_target, g_floor)

            for i, si in enumerate(species):
                for j, sj in enumerate(species):

                    if not fixed_mask[i, j]:
                        continue

                    plt.figure(figsize=(6, 4))

                    plt.plot(
                        r,
                        g_target_safe[i, j],
                        "k--",
                        lw=2,
                        label="target g(r)",
                    )

                    plt.plot(
                        r,
                        g_pred_safe[i, j],
                        "b-",
                        lw=2,
                        label="predicted g(r)",
                    )

                    plt.xlabel("r")
                    plt.ylabel("g(r)")
                    plt.title(f"RDF ({sname}): {si}{sj}")
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig(
                        plots_dir / f"{filename_prefix}_rdf_{sname}_{si}{sj}.png", dpi  = 600
                    )
                    plt.close()

        print(f"✅ Potential and RDF plots exported to: {plots_dir}")

    
    
    
    return u_matrix / beta_ref, sigma_matrix

