
# cdft_solver/generators/rdf_isotropic.py

import json
import numpy as np
from scipy.fftpack import dst, idst
from scipy.interpolate import interp1d
from collections import defaultdict
from pathlib import Path
from scipy.special import j0

from collections.abc import Mapping
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials

import matplotlib.pyplot as plt
import re

from .closure import closure_update_c_matrix




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


import matplotlib.pyplot as plt
from pathlib import Path

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





def process_supplied_rdf(supplied_data, species, r_grid):
    """
    Uses canonical pair keys like 'AA', 'AB', ...

    Returns
    -------
    g_fixed : (N, N, Nr) ndarray
    fixed_mask : (N, N) bool ndarray
    """

    N = len(species)
    Nr = len(r_grid)

    g_fixed = np.zeros((N, N, Nr))
    fixed_mask = np.zeros((N, N), dtype=bool)

    if supplied_data is None:
        return g_fixed, fixed_mask

    rdf_dict = find_key_recursive(supplied_data, "rdf")


    # --- create canonical species pairs ---
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

            g_fixed[i, j, :] = g_interp
            fixed_mask[i, j] = True

    return g_fixed, fixed_mask





import numpy as np
from numba import njit, prange

# ============================================================
# FAST NUMBA DST (Type-I equivalent)
# ============================================================

@njit(parallel=True, fastmath=True)
def dst_type1_numba(x):
    N = x.shape[0]
    X = np.zeros(N)

    for k in prange(N):
        s = 0.0
        for n in range(N):
            s += x[n] * np.sin(np.pi * (k+1) * (n+1) / (N+1))
        X[k] = s

    return X


@njit(parallel=True, fastmath=True)
def idst_type1_numba(X):
    N = X.shape[0]
    x = np.zeros(N)

    for n in prange(N):
        s = 0.0
        for k in range(N):
            s += X[k] * np.sin(np.pi * (k+1) * (n+1) / (N+1))
        x[n] = s

    return x


# ============================================================
# FAST 2D HANKEL (DST-BASED)
# ============================================================

@njit(fastmath=True)
def hankel_forward_2d_numba(f_r, r):
    N = r.shape[0]
    dr = r[1] - r[0]
    Rmax = r[-1]

    k = np.pi * (np.arange(1, N+1)) / Rmax

    x = r * f_r
    X = dst_type1_numba(x)

    Fk = (2.0 * np.pi * dr / k) * X

    return k, Fk


@njit(fastmath=True)
def hankel_inverse_2d_numba(k, Fk, r):
    N = r.shape[0]
    Rmax = r[-1]
    dk = np.pi / Rmax

    Y = k * Fk
    y = idst_type1_numba(Y)

    f_r = np.zeros_like(r)

    for i in range(N):
        if r[i] > 1e-12:
            f_r[i] = (dk / (4.0 * np.pi**2 * r[i])) * y[i]
        else:
            f_r[i] = y[1]  # regularization

    return f_r


# ============================================================
# MATRIX VERSIONS (FULLY JITTED)
# ============================================================

@njit(parallel=True, fastmath=True)
def hankel_transform_matrix_numba(f_r_matrix, r):
    N = f_r_matrix.shape[0]
    Nr = r.shape[0]

    f_k_matrix = np.zeros((N, N, Nr))

    for i in prange(N):
        for j in range(N):
            k, Fk = hankel_forward_2d_numba(f_r_matrix[i, j], r)
            f_k_matrix[i, j] = Fk

    return f_k_matrix, k


@njit(parallel=True, fastmath=True)
def inverse_hankel_transform_matrix_numba(f_k_matrix, k, r):
    N = f_k_matrix.shape[0]
    Nr = r.shape[0]

    f_r_matrix = np.zeros((N, N, Nr))

    for i in prange(N):
        for j in range(N):
            f_r_matrix[i, j] = hankel_inverse_2d_numba(k, f_k_matrix[i, j], r)

    return f_r_matrix


# ============================================================
# FULL NUMBA OZ SOLVER (NO C, FULLY FAST)
# ============================================================

@njit(fastmath=True)
def solve_oz_kspace_numba(c_k, densities):
    N, _, Nk = c_k.shape

    gamma_k = np.zeros_like(c_k)

    for ik in range(Nk):

        # Build matrices
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                C[i, j] = c_k[i, j, ik]

        # OZ matrix solve
        A = np.eye(N)
        for i in range(N):
            for j in range(N):
                A[i, j] -= densities[j] * C[i, j]

        # Solve (A^-1 - I)C
        A_inv = np.linalg.inv(A)

        for i in range(N):
            for j in range(N):
                gamma_k[i, j, ik] = 0.0
                for m in range(N):
                    gamma_k[i, j, ik] += (A_inv[i, m] - (1.0 if i == m else 0.0)) * C[m, j]

    return gamma_k


# ============================================================
# FAST OZ SOLVER (FULL REPLACEMENT)
# ============================================================

def multi_component_oz_solver_alpha(
    r,
    pair_closures,
    densities,
    u_matrix,
    sigma_matrix=None,
    n_iter=5000,
    tol=1e-8,
    alpha_max=0.05,
):

    densities = np.asarray(densities, float)
    N = pair_closures.shape[0]
    Nr = len(r)

    gamma_r = np.zeros((N, N, Nr))

    # initial closure
    c_r = closure_update_c_matrix(
        gamma_r, r, pair_closures, u_matrix, sigma_matrix
    )

    alpha = 1e-4

    for it in range(n_iter):

        gamma_old = gamma_r.copy()

        # --- closure
        c_r = closure_update_c_matrix(
            gamma_r, r, pair_closures, u_matrix, sigma_matrix
        )

        # --- transform
        c_k, k = hankel_transform_matrix_numba(c_r, r)

        # --- OZ solve
        gamma_k = solve_oz_kspace_numba(c_k, densities)

        # --- inverse transform
        gamma_new = inverse_hankel_transform_matrix_numba(gamma_k, k, r)

        # --- mixing
        gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new

        # --- convergence
        err = np.max(np.abs(gamma_r - gamma_old))

        if it % 50 == 0:
            print(f"Iter {it} | Δγ = {err:.3e} | α = {alpha:.4f}")

        if err < tol:
            print(f"✅ Converged in {it} iterations")
            break

        # adaptive alpha
        if err < 1e-2:
            alpha = min(alpha * 1.1, alpha_max)
        else:
            alpha = max(alpha * 0.7, 1e-5)

    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r
    
    
    
    
def apply_adaptive_ylim(ax, ydata, limit=10, clip=5):
    ymax = np.nanmax(np.abs(ydata))
    if ymax > limit:
        ax.set_ylim(-clip, clip)
def plot_matrix_quantity( r, quantity, u_matrix, species, title_prefix, filename, plots_dir):
    
    n = len(species)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n), sharex=True)

    if n == 1:
        axes = np.array([[axes]])

    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            ax = axes[i, j]

            y = quantity[i, j]
            ax.plot(r, y, label="value")
            ax.plot(r, u_matrix[i, j], "--", alpha=0.7, label="u(r)")

            apply_adaptive_ylim(ax, u_matrix[i, j])

            if i == n - 1:
                ax.set_xlabel("r")
            if j == 0:
                ax.set_ylabel(f"{si}")

            ax.set_title(f"{si}-{sj}", fontsize=9)
            ax.grid(alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    fig.savefig(Path(plots_dir) / filename, dpi=300)
    plt.close(fig)



# -----------------------------
# RDF core function
# -----------------------------

# cdft_solver/generators/rdf_isotropic.py

# =============================
# Utilities
# =============================

def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None




# =============================
# Main RDF driver
# =============================

def rdf_2d(
    ctx,
    rdf_config,
    densities,
    supplied_data=None,
    export=False,
    plot=True,
    filename_prefix="rdf",
):
    """
    Fully dictionary-driven isotropic RDF solver
    with optional constrained h(r) projection.
    """

    # -----------------------------
    # Extract RDF parameters
    # -----------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    if rdf_block is None:
        raise KeyError("No 'rdf' key found in rdf_config")

    #params = rdf_config["rdf_parameters"]
    species = find_key_recursive(rdf_config, "species")
    N = len(species)

    beta = rdf_block.get("beta", 1.0)
    tol = rdf_block.get("tolerance", 1e-6)
    n_iter = find_key_recursive(rdf_config, "max_iteration")
    alpha_max = rdf_block.get("alpha_max", 0.05)
    
    
    
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
    Nr =n_points
    r_max  =  rdf_block.get("r_max", 6)
    dr = r_max / (n_points + 1)
    r = dr * np.arange(1, n_points + 1)
    r_grid  = r
    
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
    n = len(species)
    u_matrix = np.zeros((n, n, len(r)))

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

            u_val = beta * interp_u(r)

            # symmetric assignment
            u_matrix[i, j, :] = u_val
            u_matrix[j, i, :] = u_val
            
    plots = Path(ctx.plots_dir)      
    plot_u_matrix(
    r=r,
    u_matrix=u_matrix,
    species=species,
    outdir=plots,
    filename="pair_potentials.png",
)

  

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
    
    
    #print(sigma_matrix)

    # ============================================================
    # STEP 1: Unconstrained OZ solve
    # ============================================================
    c_r, gamma_r, g_r = multi_component_oz_solver_alpha(
        r=r,
        pair_closures=pair_closures,
        densities=np.asarray(densities, float),
        u_matrix=u_matrix,
        sigma_matrix=sigma_matrix,
        n_iter=n_iter,
        tol=tol,
        alpha_rdf_max=alpha_max,
    )

    # ============================================================
    # STEP 2: Constrained projection loop (optional)
    # ============================================================
    if supplied_data is not None:

        g_fixed, fixed_mask = process_supplied_rdf(
            supplied_data, species, r
        )

        # Initial h
        h_r = g_r - 1.0

        for proj_iter in range(n_iter):

            h_old = h_r.copy()

            # --- Replace supplied h(r)
            for i in range(N):
                for j in range(N):
                    if fixed_mask[i, j]:
                        h_r[i, j, :] = g_fixed[i, j, :] - 1.0

            # --- Hankel transform h → k
            h_k, k = hankel_transform_matrix_fast(h_r, r)

            # --- Compute c(k) from OZ identity
            # h_k = c_k + c_k * rho * h_k  → matrix solve
            c_k = solve_oz_kspace(h_k, densities)

            # --- Back transform c(k) → c(r)
            c_r_proj = inverse_hankel_transform_matrix_fast(c_k, k, r)

            # --- Freeze supplied pairs
            c_update_flag = np.ones((N, N), dtype=bool)
            for i in range(N):
                for j in range(N):
                    if fixed_mask[i, j]:
                        c_update_flag[i, j] = False

            # --- OZ solve with frozen c
            c_r, gamma_r, g_r = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities, float),
                u_matrix=u_matrix,
                sigma_matrix=sigma_matrix,
                c_update_flag=c_update_flag,
                c_initial=c_r_proj,
                n_iter=n_iter,
                tol=tol,
                alpha_rdf_max=alpha_max,
            )

            h_r = g_r - 1.0

            # --- Convergence check (FREE pairs only)
            diff = 0.0
            for i in range(N):
                for j in range(N):
                    if not fixed_mask[i, j]:
                        diff = max(
                            diff,
                            np.max(np.abs(h_r[i, j] - h_old[i, j]))
                        )

            if diff < tol:
                print(f"✅ Projection converged in {proj_iter+1} iterations.")
                break

    # ============================================================
    # Output
    # ============================================================
    rdf_out = {}
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            rdf_out[(si, sj)] = {
                "r": r,
                "g_r": g_r[i, j],
                "c_r": c_r[i, j],
                "gamma_r": gamma_r[i, j],
                "u_r": u_matrix[i, j],
            }

    if export:
        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_out = {
            "metadata": {
                "species": species,
                "densities": list(map(float, densities)),
                "beta": float(beta),
            },
            "pairs": {}
        }

        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                pair_key = f"{si}{sj}"

                json_out["pairs"][pair_key] = {
                    "r": r.tolist(),
                    "g_r": g_r[i, j].tolist(),
                    "h_r": (g_r[i, j] - 1.0).tolist(),
                    "c_r": c_r[i, j].tolist(),
                    "gamma_r": gamma_r[i, j].tolist(),
                    "u_r": u_matrix[i, j].tolist(),
                }

        json_path = out / f"{filename_prefix}_rdf.json"

        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=4)

        print(f"✅ RDF results exported to JSON → {json_path}")

    # -----------------------------
    # Plotting
    # -----------------------------
    if plot:
        plots = Path(ctx.plots_dir)
        plots.mkdir(parents=True, exist_ok=True)

        # Convert densities array to a filename-friendly string
        rho_str = "_".join(f"{rho:.3f}" for rho in densities)

        plot_matrix_quantity(
            r, g_r, u_matrix, species,
            title_prefix="g(r)",
            filename=f"{filename_prefix}_gr_matrix_rho_{rho_str}.png",
            plots_dir=plots
        )

        plot_matrix_quantity(
            r, c_r, u_matrix, species,
            title_prefix="c(r)",
            filename=f"{filename_prefix}_cr_matrix_rho_{rho_str}.png",
            plots_dir=plots
        )

        plot_matrix_quantity(
            r, gamma_r, u_matrix, species,
            title_prefix="γ(r)",
            filename=f"{filename_prefix}_gammar_matrix_rho_{rho_str}.png",
            plots_dir=plots
        )


    return rdf_out
