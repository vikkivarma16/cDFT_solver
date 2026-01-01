# cdft_solver/generators/rdf_isotropic.py

import json
import numpy as np
from scipy.fftpack import dst, idst
from scipy.interpolate import interp1d
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import re
from .closer import closure_update_c_matrix
from scipy.optimize import minimize_scalar




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



def process_supplied_rdf_multistate(supplied_data, species, r_grid):
    
    """
    Process multistate supplied RDF data.
    Each state must define:
      - densities: array-like (N,)
      - temperature or beta
      - rdf: dict with pair keys like 'AA', 'AB', ...
    Returns
    -------
    states : dict
        states[state_name] = {
            "densities": (N,) ndarray,
            "beta": float,
            "g_target": (N, N, Nr) ndarray,
            "fixed_mask": (N, N) bool ndarray,
        }
    """

    if supplied_data is None:
        return {}

    # Locate state container
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
        if "densities" not in state_data:
            raise KeyError(f"State '{state_name}' missing 'densities'")
        densities = np.asarray(state_data["densities"], dtype=float)

        if len(densities) != N:
            raise ValueError(
                f"State '{state_name}' densities size mismatch: "
                f"expected {N}, got {len(densities)}"
            )

        if "beta" in state_data:
            beta = float(state_data["beta"])
        elif "temperature" in state_data:
            beta = 1.0 / float(state_data["temperature"])
        else:
            beta = 1.0  # default

        rdf_dict = find_key_recursive(state_data, "rdf")
        if rdf_dict is None:
            raise KeyError(f"State '{state_name}' has no RDF data")

        # -----------------------------
        # Allocate arrays
        # -----------------------------
        g_target = np.zeros((N, N, Nr))
        fixed_mask = np.zeros((N, N), dtype=bool)

        # -----------------------------
        # Read RDF pairs
        # -----------------------------
        for i, si in enumerate(species):
            for j, sj in enumerate(species):

                pair_key = f"{si}{sj}"
                if pair_key not in rdf_dict:
                    continue

                entry = rdf_dict[pair_key]

                r_sup = np.asarray(entry["r"], dtype=float)
                g_sup = np.asarray(entry["g"], dtype=float)

                interp = interp1d(
                    r_sup,
                    g_sup,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(g_sup[0], g_sup[-1]),
                )

                g_interp = interp(r_grid)

                # Enforce symmetry
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
    grid_dict,
    supplied_data,
    sigma=None,
    potential_dict=None,
    export=False,
    filename_prefix="boltzmann",
):
    """
    Multistate Isotropic Boltzmann / IBI inversion
    using OZ + closure.
    """

    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    params = rdf_config["rdf_parameters"]

    species = params["species"]
    N = len(species)

    tol = rdf_block.get("tolerance", 1e-6)
    max_iter = rdf_block.get("max_iteration", 200)
    alpha_ibi = rdf_block.get("alpha_ibi", 0.1)

    g_floor = 1e-8
    hard_core_repulsion = 1e6

    # -------------------------------------------------
    # r grid
    # -------------------------------------------------
    r = np.linspace(
        grid_dict["r_min"],
        grid_dict["r_max"],
        grid_dict["n_points"],
    )
    Nr = len(r)

    # -------------------------------------------------
    # Closures
    # -------------------------------------------------
    closure_cfg = rdf_block["closure"]
    pair_closures = np.empty((N, N), dtype=object)
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            pair_closures[i, j] = closure_cfg[f"{si}{sj}"]

    # -------------------------------------------------
    # Multistate RDF
    # -------------------------------------------------
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
    u_matrix = np.zeros((N, N, Nr))
    invert_mask = np.zeros((N, N), dtype=bool)
    
    if potential_dict is not None:
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                key = (si, sj)
                rkey = (sj, si)

                pdata = potential_dict.get(key, potential_dict.get(rkey))
                if pdata is None:
                    raise KeyError(f"Missing potential for pair {si}-{sj}")

                interp_u = interp1d(
                    pdata["r"], pdata["u"],
                    bounds_error=False,
                    fill_value=0.0
                )
                u_matrix[i, j, :] = beta_ref*interp_u(r)

    

    # Initialize from first state RDF
    s0 = state_names[0]
    g0 = states[s0]["g_target"]
    
    enable_sigma_refinement = 0 
    for sname, sdata in states.items():
        mask = sdata["fixed_mask"]
        for i in range(N):
            for j in range(N):
                if (mask[i, j] == True):
                    g_safe = np.maximum(g0[i, j], g_floor)
                    u_matrix[i, j] = u_matrix[j, i] = -np.log(g_safe)
                    sigma_matrix[i, j] = sigma_matrix[j, i] = detect_sigma_from_gr(r, g0[i, j])
                    invert_mask[i, j] = invert_mask[j, i] = True
                    if (sigma_matrix[i, j] > 0.0):
                        enable_sigma_refinement =  1

    sigma_update_every = 5
    sigma_freeze_after = 50
    # -------------------------------------------------
    # Multistate IBI loop
    # -------------------------------------------------
    for it in range(1, max_iter + 1):

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

