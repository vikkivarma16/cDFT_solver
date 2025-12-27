# cdft_solver/generators/rdf_isotropic.py

import json
import numpy as np
from scipy.fftpack import dst, idst
from scipy.interpolate import interp1d
from collections import defaultdict
from pathlib import Path

# Import project-specific splitters / generator
from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials
from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi
from cdft_solver.generators.potential_splitter.generator_potential_total import raw_potentials

import matplotlib.pyplot as plt
import re


# -----------------------------
# DST-based Hankel forward/inverse transforms
# -----------------------------

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

def closure_update_c_matrix(gamma_r_matrix, r, beta, pair_closures, u_soft_matrix, sigma_matrix=None):
    N = gamma_r_matrix.shape[0]
    c_new = np.zeros_like(gamma_r_matrix)
    for a in range(N):
        for b in range(N):
            closure = pair_closures[a, b].upper()
            u = u_soft_matrix[a, b, :]
            gamma = gamma_r_matrix[a, b, :]
            if closure == "PY":
                sigma_ab = sigma_matrix[a, b] if sigma_matrix is not None else 1.0
                core_mask = r < sigma_ab
                c_in = -(1 + gamma)
                c_out = (1.0 + gamma) * (np.exp(-beta * u) - 1.0)
                c_new[a, b, :] = np.where(core_mask, c_in, c_out)
            elif closure == "HNC":
                c_new[a, b, :] = np.exp(-beta * u + gamma) - gamma - 1.0
            elif closure == "HYBRID":
                sigma_ab = sigma_matrix[a, b] if sigma_matrix is not None else 1.0
                core_mask = r < sigma_ab
                c_in = -(1 + gamma)
                c_out = np.exp(-beta * u + gamma) - gamma - 1.0
                c_new[a, b, :] = np.where(core_mask, c_in, c_out)
    return c_new


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


def multi_component_oz_solver_alpha(r, pair_closures, densities, beta, sigma_matrix=None, u_soft_matrix=None, n_iter=10000, tol=1e-12, alpha_rdf_max=0.1):
    """
    Multi-component Ornstein–Zernike (OZ) solver with adaptive alpha-mixing.

    Parameters
    ----------
    r : np.ndarray
        Radial grid points.
    pair_closures : np.ndarray
        N×N matrix of closure types (e.g., "HNC", "PY").
    densities : np.ndarray
        Species densities.
    beta : float
        Inverse temperature (1/kT).
    sigma_matrix : np.ndarray, optional
        Effective diameter matrix.
    u_soft_matrix : np.ndarray
        Soft potential matrix, required.
    n_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    alpha_rdf_max : float
        Maximum allowed alpha for adaptive mixing.

    Returns
    -------
    c_r : np.ndarray
        Direct correlation functions.
    gamma_r : np.ndarray
        Indirect correlation functions.
    g_r : np.ndarray
        Radial distribution functions.
    """

    if u_soft_matrix is None:
        raise ValueError("u_soft_matrix must be provided.")

    N = pair_closures.shape[0]
    gamma_r = np.zeros_like(u_soft_matrix)
    c_r = np.zeros_like(u_soft_matrix)

    print(f"\n🚀 Starting OZ solver (adaptive α, α_max = {alpha_rdf_max})")
    print(f"{'Iter':>6s} | {'Δγ(max)':>12s} | {'α':>6s}")

    prev_diff = np.inf
    alpha = min(0.01, alpha_rdf_max)  # start small

    for step in range(n_iter):
        # Update closure relation
        c_r = closure_update_c_matrix(gamma_r, r, beta, pair_closures, u_soft_matrix, sigma_matrix)

        # Solve OZ equation
        gamma_new = solve_oz_matrix(c_r, r, densities)

        # Compute difference
        delta_gamma = gamma_new - gamma_r
        diff = np.max(np.abs(delta_gamma))

        # Adaptive alpha: increase if converging fast, decrease if oscillating
        if diff < prev_diff:
            alpha = min(alpha * 1.05, alpha_rdf_max)
        else:
            alpha = max(alpha * 0.5, 1e-4)

        # Mix update
        gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new

        # Print progress every 10 iterations or on convergence
        if step % 10 == 0 or diff < tol:
            print(f"{step:6d} | {diff:12.3e} | {alpha:6.4f}")

        # Check convergence
        if diff < tol:
            print(f"\n✅ Converged in {step+1} iterations (Δγ = {diff:.3e}, α = {alpha:.4f}).")
            break

        prev_diff = diff

    else:
        print(f"\n⚠️ Warning: not converged after {n_iter} iterations (Δγ = {diff:.3e}).")

    # Compute final correlation functions
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r




# -----------------------------
# RDF core function
# -----------------------------

def rdf(species,
        pair_closures,
        densities,
        r,
        beta=1.0,
        n_iter=10000,
        tol=1e-6,
        pre_u_matrix=None,
        pre_sigma_matrix=None, alpha_rdf_max=0.01):
    """Compute c_r, gamma_r, g_r, and vk."""
    alpha =  alpha_rdf_max 
    N = len(species)
    if pre_u_matrix is None or pre_sigma_matrix is None:
        raise ValueError("Please provide pre_u_matrix and pre_sigma_matrix to rdf().")

    c_r, gamma_r, g_r = multi_component_oz_solver_alpha(
        r, pair_closures, densities, beta,
        sigma_matrix=pre_sigma_matrix, u_soft_matrix=pre_u_matrix,
        n_iter=n_iter, tol=tol, alpha_rdf_max = alpha)

    
    return c_r, gamma_r, g_r


# -----------------------------
# Entry function rdf_isotropic
# -----------------------------

def rdf_isotropic(ctx):
    """
    Top-level RDF driver that reads density & grid parameters
    from JSON and runs the isotropic RDF solver.
    """

    # -----------------------------
    # Read JSON input file
    # -----------------------------
    scratch = Path(ctx.scratch_dir)
    json_file = scratch / "input_data_iso_rdf.json"
    if not json_file.exists():
        raise FileNotFoundError(f"RDF input JSON not found: {json_file}")

    with open(json_file, "r") as f:
        params = json.load(f)
        params_new  = params["rdf_parameters"]
        

    densities_dict = params.get("densities", {})
    densities = np.array(list(densities_dict.values()), dtype=float)
    
    r_max = float(params_new.get("r_max", 5.0))
    n_points = int(params_new.get("n_points", 400))
    r_min = float(params_new.get("r_min", r_max/n_points))
    beta = float(params_new.get("beta", 1.0))
    alpha_rdf_max = float(params_new.get("alpha_rdf_max", 0.1))
    

    print("\nLoaded RDF parameters:")
    print(json.dumps(params, indent=4))

    # Build r grid
    r = np.linspace(r_min, r_max, n_points)

    # Load potentials
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)

    species = pot_data["species"]
    interactions_by_level = pot_data["interactions"] or raw_potentials
    levels = ["primary", "secondary", "tertiary"]
    N = len(species)
    Nr = len(r)

    # sigma matrix (Lorentz rule)
    if hc_data is None:
        sigma_eff = np.zeros(N)
        flags = np.zeros(N, dtype=int)
    else:
        sigma_eff = np.array([hc_data.get(sp, {}).get("sigma_eff", 0.0) for sp in species])
        flags = np.array([hc_data.get(sp, {}).get("flag", 0) for sp in species])

    sigma_matrix = 0.5 * (sigma_eff[:, None] + sigma_eff[None, :])
    flag_mask = (flags[:, None] == 0) & (flags[None, :] == 0)
    sigma_matrix[flag_mask] = 0.0

    # Build potential matrix
    u_matrix = np.zeros((N, N, Nr))
    for level in levels:
        level_dict = interactions_by_level.get(level, {}) or {}
        for pair_key, params in level_dict.items():
            if len(pair_key) != 2:
                continue
            a, b = pair_key
            if a not in species or b not in species:
                continue

            i, j = species.index(a), species.index(b)
            V_func = ppi(params)
            r_space = np.linspace(0.0, r_max, Nr)
            V_r_local = V_func(r_space)
            interp = interp1d(r_space, V_r_local, bounds_error=False, fill_value="extrapolate")
            V_r = interp(r)
            u_matrix[i, j, :] += V_r
            if i != j:
                u_matrix[j, i, :] += V_r

    # closure assignment
    pair_closures = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            pair_closures[i, j] = "HNC" if sigma_matrix[i, j] == 0 else "PY"
            
            
    print ("---------- Please print the closures -----------\n\n", pair_closures)

    # Run RDF
    c_r, gamma_r, g_r = rdf(
        species=species,
        pair_closures=pair_closures,
        densities=densities,
        r=r,
        beta=beta,
        pre_u_matrix=u_matrix,
        pre_sigma_matrix=sigma_matrix, alpha_rdf_max = alpha_rdf_max,
    )



    # -----------------------------
    # Save RDF (g_r) and c_r results
    # -----------------------------
    out_gr_txt = scratch / "rdf_gr_columns.txt"
    out_cr_txt = scratch / "rdf_cr_columns.txt"
    out_gamma_r_txt = scratch / "rdf_gamma_r_columns.txt"

    # Prepare header with pair labels
    pair_labels = []
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            pair_labels.append(f"{sp_i}-{sp_j}")

    # --- Save g(r) ---
    with open(out_gr_txt, "w") as f:
        f.write("# Radial Distribution Functions (g_r)\n")
        f.write("# Columns: r " + " ".join([f"g_{p}" for p in pair_labels]) + "\n")
        f.write("# beta = %.6f\n" % beta)
        f.write("# r_min = %.6f, r_max = %.6f, n_points = %d\n\n" % (r_min, r_max, n_points))

        for k in range(len(r)):
            row = [f"{r[k]:.6e}"] + [f"{g_r[i,j,k]:.6e}" for i in range(len(species)) for j in range(len(species))]
            f.write(" ".join(row) + "\n")

    print(f"✅ g(r) results written to {out_gr_txt}")

    # --- Save c(r) ---
    with open(out_cr_txt, "w") as f:
        f.write("# Direct Correlation Functions (c_r)\n")
        f.write("# Columns: r " + " ".join([f"c_{p}" for p in pair_labels]) + "\n")
        f.write("# beta = %.6f\n" % beta)
        f.write("# r_min = %.6f, r_max = %.6f, n_points = %d\n\n" % (r_min, r_max, n_points))

        for k in range(len(r)):
            row = [f"{r[k]:.6e}"] + [f"{c_r[i,j,k]:.6e}" for i in range(len(species)) for j in range(len(species))]
            f.write(" ".join(row) + "\n")

    print(f"✅ c(r) results written to {out_cr_txt}")
    
    
    
    with open(out_gamma_r_txt, "w") as f:
        f.write("# InDirect Correlation Functions (gamma_r)\n")
        f.write("# Columns: r " + " ".join([f"c_{p}" for p in pair_labels]) + "\n")
        f.write("# beta = %.6f\n" % beta)
        f.write("# r_min = %.6f, r_max = %.6f, n_points = %d\n\n" % (r_min, r_max, n_points))

        for k in range(len(r)):
            row = [f"{r[k]:.6e}"] + [f"{gamma_r[i,j,k]:.6e}" for i in range(len(species)) for j in range(len(species))]
            f.write(" ".join(row) + "\n")

    print(f"✅ c(r) results written to hi {out_gamma_r_txt}")



    # -----------------------------
    # Plot g(r)
    # -----------------------------
        # -----------------------------
    # Plot g(r) and u(r)
    # -----------------------------
    plots_dir = Path(ctx.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    densities_str = "_".join(f"{d:.4f}" for d in densities)
    densities_str = re.sub(r"[^0-9A-Za-z_.-]", "_", densities_str)

    
    # --- g(r) + u(r) plot ---
    fig, axes = plt.subplots(N, N, figsize=(4*N, 3.5*N), dpi=120, sharex=True, sharey=True)
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            ax = axes[i, j] if N > 1 else axes
            ax.plot(r, g_r[i, j, :], color='tab:blue', lw=1.8, label="g(r)")
            ax.set_title(f"{sp_i}-{sp_j}", fontsize=10)
            ax.set_xlim(r[0], r[-1])
            ax.grid(alpha=0.3)

            # Plot u(r) on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(r, u_matrix[i, j, :], color='tab:red', lw=1.2, ls='--', label="u(r)")

    plt.suptitle("Pair Correlation g(r) and Potential u(r)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = plots_dir / f"rdf_all_pairs_{densities_str}.png"
    plt.savefig(out_path, dpi=600)
    plt.close()
    print(f"📊 Saved RDF plot grid → {out_path}")

    # -----------------------------
    # Plot -log(g(r)) and u(r) on same y-axis
    # -----------------------------
    fig, axes = plt.subplots(N, N, figsize=(4*N, 3.5*N), dpi=120, sharex=True, sharey=True)

    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            ax = axes[i, j] if N > 1 else axes
            g_vals = g_r[i, j, :]
            safe_g = np.where(g_vals > 1e-12, g_vals, 1e-12)

            # Plot -log(g(r))
            ax.plot(r, -np.log(safe_g), color='tab:green', lw=1.8, label='-log(g(r))')

            # Plot u(r) on same y-axis
            ax.plot(r, u_matrix[i, j, :], color='tab:red', lw=1.2, ls='--', label='u(r)')

            # Formatting
            ax.set_title(f"{sp_i}-{sp_j}", fontsize=10)
            ax.set_xlim(r[0], r[-1])
            ax.grid(alpha=0.3)

            # Add legend only once per subplot
            ax.legend(fontsize=8, loc='best', frameon=False)

    plt.suptitle("Effective Potential Comparison: -log(g(r)) and u(r)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path_log = plots_dir / f"rdf_minuslog_gr_sameaxis_{densities_str}.png"
    plt.savefig(out_path_log, dpi=600)
    plt.close()
    print(f"📊 Saved -log(g(r)) + u(r) (same scale) plot grid → {out_path_log}")




    # No return value; data is written to file
    return g_r 
