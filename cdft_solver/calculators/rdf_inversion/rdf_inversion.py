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

from collections.abc import Mapping
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials

from cdft_solver.calculators.radial_distribution_function.closure import closure_update_c_matrix


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
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    print(f"✅ u(r) matrix plot saved to: {outpath}")


def optimize_sigma_single_pair(
    r,
    g_target_ij,
    u_matrix,
    sigma_matrix,
    pair_closures,
    densities,
    beta,
    pair_index,
    sigma_bounds=(0.8, 1.2),
    fit_r_factor=1.3,
    sigma_relax=0.3,
    oz_n_iter=300,
    oz_tol=1e-6,
):
    """
    Optimize sigma_ij by minimizing ||g_pred - g_target||^2
    near contact using the FULL multicomponent OZ solver.
    """

    i, j = pair_index

    sigma_old = sigma_matrix[i, j]
    if sigma_old <= 0:
        return sigma_old

    # Only fit close to contact
    r_max = fit_r_factor * sigma_old
    mask = r <= r_max

    def loss(sigma_trial):
        sigma_tmp = sigma_matrix.copy()
        sigma_tmp[i, j] = sigma_trial
        sigma_tmp[j, i] = sigma_trial  # enforce symmetry

        _, _, g_pred = multi_component_oz_solver_alpha(
            r=r,
            pair_closures=pair_closures,
            densities=densities,
            u_matrix=u_matrix,
            sigma_matrix=sigma_tmp,
            n_iter=oz_n_iter,
            tol=oz_tol,
            alpha_rdf_max=0.05,
        )

        diff = g_pred[i, j, mask] - g_target_ij[mask]
        return np.mean(diff * diff)

    res = minimize_scalar(
        loss,
        bounds=(sigma_bounds[0] * sigma_old, sigma_bounds[1] * sigma_old),
        method="bounded",
    )

    sigma_new = (
        (1.0 - sigma_relax) * sigma_old
        + sigma_relax * res.x
    )

    return sigma_new



def optimize_sigma_multistate(
    r,
    u_matrix,
    sigma_matrix,
    pair_closures,
    states,
    pair_index,
    w_state,
    beta_ref,
    sigma_bounds=(0.8, 1.2),
):
    """
    Multistate sigma optimization by aggregating
    single-state sigma optimizations.
    """

    i, j = pair_index

    

    sigma_old = sigma_matrix[i, j]
    if sigma_old <= 0:
        return sigma_old

    sigma_proposals = []
    weights = []

    for sname, sdata in states.items():

        g_target = sdata["g_target"][i, j]
        densities = sdata["densities"]
        beta_s = sdata["beta"]

        # Map state beta → reference beta
        beta_eff = beta_ref

        sigma_s = optimize_sigma_single_pair(
            r=r,
            g_target_ij=g_target,
            u_matrix=u_matrix,
            sigma_matrix=sigma_matrix,
            pair_closures=pair_closures,
            densities=densities,
            beta=beta_eff,
            pair_index=(i, j),
            sigma_bounds=sigma_bounds,
        )

        sigma_proposals.append(sigma_s)
        weights.append(w_state[sname])

    # Weighted average of sigma proposals
    sigma_new = np.sum(
        w * s for w, s in zip(weights, sigma_proposals)
    ) / np.sum(weights)

    return sigma_new


import numpy as np
from scipy.interpolate import interp1d

def process_supplied_rdf_multistate(supplied_data, species, r_grid):
    """
    Process multistate supplied RDF data.

    Each state in supplied_data must define:
      - 'densities': array-like (N,)
      - 'temperature' or 'beta' (optional, default beta=1.0)
      - 'rdf': dict with pair keys like 'AA', 'AB', etc.

    Parameters
    ----------
    supplied_data : dict
        Already materialized supplied data (files loaded into x/y arrays)
    species : list of str
        List of species names
    r_grid : ndarray
        Grid points where RDFs are interpolated

    Returns
    -------
    states : dict
        states[state_name] = {
            "densities": (N,) ndarray,
            "beta": float,
            "g_target": (N, N, Nr) ndarray,
            "fixed_mask": (N, N) bool ndarray
        }
    """

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

        # Convert to ndarray, handle single scalar
        densities = np.atleast_1d(np.asarray(densities_raw, dtype=float))
        print (densities)
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




def hankel_forward_dst(f_r, r):
    """Forward 3D Hankel transform using DST-I mapping."""
    N = len(r)
    dr = r[1] - r[0]
    Rmax = (N + 1) * dr
    k = np.pi * np.arange(1, N + 1) / Rmax

    x = r * f_r
    X = dst(x, type=1)
    Fk = (2.0 * np.pi * dr / k) * X
    return k, Fk


def hankel_inverse_dst(k, Fk, r):
    """Inverse 3D Hankel transform using IDST-I mapping."""
    N = len(r)
    dr = r[1] - r[0]
    Rmax = (N + 1) * dr
    dk = np.pi / Rmax

    Y = k * Fk
    y = idst(Y, type=1)
    f_r = (dk / (4.0 * np.pi**2 * r)) * y
    return f_r


def hankel_transform_matrix_fast(f_r_matrix, r):
    N = f_r_matrix.shape[0]
    f_k_matrix = np.zeros_like(f_r_matrix)
    k = None
    for a in range(N):
        for b in range(N):
            k, Fk = hankel_forward_dst(f_r_matrix[a, b, :], r)
            f_k_matrix[a, b, :] = Fk
    return f_k_matrix, k


def inverse_hankel_transform_matrix_fast(f_k_matrix, k, r):
    N = f_k_matrix.shape[0]
    f_r_matrix = np.zeros_like(f_k_matrix)
    for a in range(N):
        for b in range(N):
            f_r_matrix[a, b, :] = hankel_inverse_dst(k, f_k_matrix[a, b, :], r)
    return f_r_matrix


# -----------------------------
# Closures and OZ solver
# -----------------------------

def solve_oz_matrix(c_r_matrix, r, densities):
    N = c_r_matrix.shape[0]
    c_k_matrix, k = hankel_transform_matrix_fast(c_r_matrix, r)
    gamma_k_matrix = np.zeros_like(c_k_matrix)
    rho_matrix = np.diag(densities)
    I = np.identity(N)
    eps_reg = 1e-12

    for ik in range(len(k)):
        Ck = c_k_matrix[:, :, ik]
        num = Ck @ rho_matrix @ Ck
        A = I - Ck @ rho_matrix + eps_reg * I
        gamma_k_matrix[:, :, ik] = np.linalg.solve(A, num)

    gamma_r_matrix = inverse_hankel_transform_matrix_fast(gamma_k_matrix, k, r)
    return gamma_r_matrix


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

        if step % 10 == 0 or diff < tol:
            print(f"{step:6d} | {diff:12.3e} | {alpha:6.4f}")

        if diff < tol:
            print(f"\n✅ Converged in {step+1} iterations.")
            break

        prev_diff = diff

    else:
        print(f"\n⚠️ Warning: not converged after {n_iter} iterations.")

    # -----------------------------
    # Final observables
    # -----------------------------
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r



def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None

def detect_sigma_from_gr(r, g, g_tol=1e-6):
    """
    Detect hard-core diameter from g(r).
    Returns sigma or 0.0 if no hard core detected.
    """
    idx = np.where(g > g_tol)[0]
    if len(idx) == 0:
        return 0.0
    return r[idx[0]]
    
    
def boltzmann_potential_from_gr(g, beta=1.0, g_min=1e-8):
    """
    u(r) = -ln g(r) with numerical protection
    """
    g_safe = np.maximum(g, g_min)
    return -np.log(g_safe) / beta




    
    
    
    
def boltzmann_inversion(
    ctx,
    rdf_config,
    supplied_data,
    export=False,
    filename_prefix="boltzmann",
):
    """
    Multistate Isotropic Boltzmann / IBI inversion
    using OZ + closure.
    """

    g_floor = 1e-8
    hard_core_repulsion = 1e6

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
    tol = rdf_block.get("tolerance", 1e-6)
    n_iter = find_key_recursive(rdf_config, "max_iteration")
    alpha_max = rdf_block.get("alpha_max", 0.05)
    alpha_ibi_max = rdf_block.get("alpha_ibi_max", 0.05)
    
    
    
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
    print (pair_closures)
    N = len(species)
    n = len (species)
    
    
    states = process_supplied_rdf_multistate(
        supplied_data, species, r
    )
    if not states:
        raise ValueError("No multistate RDF data provided")

    state_names = list(states.keys())
    beta_ref = states[state_names[0]]["beta"]

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
    plot_u_matrix( r=r, u_matrix=u_matrix, species=species, outdir=plots, filename="pair_potentials.png",)

  

    # u_matrix: (N, N, Nr), r: (Nr,)
    u_strength = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            u = u_matrix[i, j, :]
            integrand = r**2 * u
            u_strength[i, j] = 4.0 * np.pi * np.trapz(integrand, r)

    #print("Integrated potential strength (trapezoidal) for each pair:")
    #print(u_strength)



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

    sigma_update_every = 5
    sigma_freeze_after = 50
    
    
    alpha_ibi = 0.01
    
    # -------------------------------------------------
    # Multistate IBI loop
    # -------------------------------------------------
    for it in range(1, n_iter + 1):

        delta_u_accum = np.zeros_like(u_matrix)
        max_diff = 0.0

        # Store per-state OZ results if sigma refinement is enabled
        # -----------------------------
        # State loop
        # -----------------------------
        for sname, sdata in states.items():

            beta_s = sdata["beta"]
            densities_s = sdata["densities"]
            g_target = sdata["g_target"]
            fixed_mask = sdata["fixed_mask"]

            c_r, gamma_r, g_pred = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=densities_s,
                u_matrix=beta_s*u_matrix/beta_ref,
                sigma_matrix=sigma_matrix,
                n_iter=1500,
                tol=tol,
                alpha_rdf_max=0.1,
            )
        
            


            g_pred_safe = np.maximum(g_pred, g_floor)
            g_target_safe = np.maximum(g_target, g_floor)

            # Cache results for sigma optimization
           

            for i in range(N):
                for j in range(N):

                    if not fixed_mask[i, j]:
                        continue



                    mask_r = g_target_safe[i, j] > 1e-4
                    delta_s = np.zeros_like(r)
                    delta_s[mask_r] = (beta_ref / beta_s) * np.log(
                        g_pred_safe[i, j, mask_r] / g_target_safe[i, j, mask_r]
                    )

                    delta_u_accum[i, j] += w_state[sname] * delta_s

                    max_diff = max(
                        max_diff,
                        np.max(np.abs(g_pred[i, j] - g_target[i, j]))
                    )

        # -----------------------------
        # Apply combined potential update
        # -----------------------------
        for i in range(N):
            for j in range(N):

                if not invert_mask[i, j]:
                    continue

                u_matrix[i, j] += alpha_ibi * delta_u_accum[i, j]

                if sigma_matrix[i, j] > 0:
                    core = r < sigma_matrix[i, j]
                    u_matrix[i, j, core] = u_matrix[j, i, core] = hard_core_repulsion

        # -----------------------------
        # Sigma refinement (ONCE per iteration)
        # -----------------------------
        # -----------------------------
        # Sigma refinement (ONCE per iteration)
        # -----------------------------
        if (enable_sigma_refinement and it % sigma_update_every == 0 and it < sigma_freeze_after):
            for i in range(N):
                for j in range(i, N):

                    if not invert_mask[i, j]:
                        continue

                    sigma_new = optimize_sigma_multistate(
                        r=r,
                        u_matrix=u_matrix,
                        sigma_matrix=sigma_matrix,
                        pair_closures=pair_closures,
                        states=states,
                        pair_index=(i, j),
                        w_state=w_state,
                        beta_ref=beta_ref,
                    )

                    sigma_matrix[i, j] = sigma_matrix[j, i] = sigma_new


        if max_diff < tol:
            print(f"\n✅ Multistate IBI converged in {it} iterations.")
            break

    else:
        print("\n⚠️ Multistate IBI did not converge.")


    # -------------------------------------------------
    # Export
    # -------------------------------------------------
    if export:
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

        with open(out / f"{filename_prefix}_potential.json", "w") as f:
            json.dump(data, f, indent=4)

        print("✅ Multistate inverted potential exported.")

    return u_matrix / beta_ref, sigma_matrix

