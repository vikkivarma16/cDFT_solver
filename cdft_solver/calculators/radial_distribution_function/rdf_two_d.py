
# cdft_solver/generators/rdf_isotropic.py

import json
import numpy as np
from scipy.fftpack import dst, idst
from scipy.interpolate import interp1d
from collections import defaultdict
from pathlib import Path

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





def hankel_forward_dst(f_r, r):
    N = len(r)
    dr = r[1] - r[0]
    Rmax = max(r)
    
    #print (Rmax)
    k = np.pi * np.arange(1, N + 1) / Rmax
    x = r * f_r
    X = dst(x, type=1)
    Fk = (2.0 * np.pi * dr / k) * X
    return k, Fk

def hankel_inverse_dst(k, Fk, r):
    N = len(r)
    dr = r[1] - r[0]
    Rmax = max (r)
    dk = np.pi / Rmax
    Y = k * Fk
    y = idst(Y, type=1)
    f_r = np.zeros_like(r)
    nonzero = ~np.isclose(r, 0.0)
    f_r[nonzero] = (dk / (4.0 * np.pi**2 * r[nonzero])) * y[nonzero]
    if np.isclose(r[0], 0.0) and len(r) > 1:
        f_r[0] = f_r[1]
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

import os
import sys
import ctypes
import numpy as np
from ctypes import c_double, c_int, POINTER

# -------------------------------
# Locate shared library reliably
# -------------------------------
_here = os.path.dirname(__file__)

if sys.platform == "darwin":
    _libname = "liboz_cylindrical.dylib"
elif sys.platform == "win32":
    _libname = "liboz_cylindrical.dll"
else:
    _libname = "liboz_cylindrical.so"

_lib_path = os.path.join(_here, _libname)

if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Shared library not found: {_lib_path}")

# Load shared library
lib = ctypes.CDLL(_lib_path)

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

lib.solve_oz_matrix.restype = None


# --------------------------------------------------
# Python wrapper (FIXED + SAFE)
# --------------------------------------------------
def solve_oz_matrix(c_r, r, densities):
    """
    Calls optimized C OZ solver (2D Hankel-based)
    """

    c_r = np.asarray(c_r, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    densities = np.asarray(densities, dtype=np.float64)

    if c_r.ndim != 3:
        raise ValueError("c_r must be shape (N, N, Nr)")

    N, N2, Nr = c_r.shape
    if N != N2:
        raise ValueError("c_r must be square in first two dims")

    # Allocate output
    gamma_r = np.zeros_like(c_r, dtype=np.float64)

    # Ensure C-contiguous memory (CRITICAL)
    c_r_flat = np.ascontiguousarray(c_r).ravel()
    gamma_r_flat = np.ascontiguousarray(gamma_r).ravel()

    r_c = np.ascontiguousarray(r)
    dens_c = np.ascontiguousarray(densities)

    # Call C
    lib.solve_oz_matrix(
        c_int(N),
        c_int(Nr),
        r_c.ctypes.data_as(POINTER(c_double)),
        dens_c.ctypes.data_as(POINTER(c_double)),
        c_r_flat.ctypes.data_as(POINTER(c_double)),
        gamma_r_flat.ctypes.data_as(POINTER(c_double)),
    )

    # Reshape back
    gamma_r = gamma_r_flat.reshape((N, N, Nr))

    # Stabilization
    gamma_r = np.clip(gamma_r, -50.0, 50.0)

    return gamma_r


# --------------------------------------------------
# Optional NumPy fallback (debug only)
# --------------------------------------------------
def hankel_forward_2d_all(c_r, J0, r_weight):
    return 2 * np.pi * np.einsum(
        'kr,abr->abk',
        J0,
        c_r * r_weight
    )


def hankel_inverse_2d_all(gamma_k, J0, k_weight):
    return (1 / (2*np.pi)) * np.einsum(
        'rk,abk->abr',
        J0,
        gamma_k * k_weight
    )


# --------------------------------------------------
# Main OZ Solver (adaptive mixing)
# --------------------------------------------------
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
    alpha_rdf_max=0.01,
):
    """
    Multi-component OZ solver with adaptive alpha-mixing
    and selective c_ij update control.
    """

    if u_matrix is None:
        raise ValueError("u_matrix must be provided.")

    densities = np.asarray(densities, dtype=float)

    N = pair_closures.shape[0]
    Nr = len(r)

    # -----------------------------
    # Initialize
    # -----------------------------
    gamma_r = np.zeros((N, N, Nr))

    if c_initial is not None:
        c_r = c_initial.copy()
    else:
        c_r = closure_update_c_matrix(
            gamma_r, r, pair_closures, u_matrix, sigma_matrix
        )

    # -----------------------------
    # Update mask
    # -----------------------------
    if c_update_flag is None:
        c_update_flag = np.ones((N, N), dtype=bool)
    else:
        c_update_flag = np.asarray(c_update_flag, dtype=bool)

    print("updating flag:", c_update_flag)

    print(f"\n🚀 Starting OZ solver (adaptive α, α_max = {alpha_rdf_max})")
    print(f"{'Iter':>6s} | {'Δγ(max)':>12s} | {'α':>8s}")

    # -----------------------------
    # Mixing setup
    # -----------------------------
    prev_diff = np.inf
    alpha = min(1e-4, alpha_rdf_max)

    # -----------------------------
    # Iteration loop
    # -----------------------------
    for step in range(n_iter):

        # --- Closure update
        c_trial = closure_update_c_matrix(
            gamma_r, r, pair_closures, u_matrix, sigma_matrix
        )

        # selective update
        for i in range(N):
            for j in range(N):
                if c_update_flag[i, j]:
                    c_r[i, j, :] = c_trial[i, j, :]

        # stabilize
        c_r = np.clip(c_r, -50.0, 50.0)

        # --- Solve OZ (C backend)
        gamma_new = solve_oz_matrix(c_r, r, densities)

        # --- Convergence
        delta_gamma = gamma_new - gamma_r
        diff = np.max(np.abs(delta_gamma))

        # --- Adaptive mixing
        gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new
        gamma_r = np.clip(gamma_r, -50.0, 50.0)

        # --- Adapt alpha
        if step % 50 == 0 or diff < tol:

            if diff < prev_diff:
                alpha = min(alpha * 1.05, alpha_rdf_max)
            else:
                alpha = max(alpha * 0.95, 0.0001)

            print(f"{step:6d} | {diff:12.3e} | {alpha:8.5f}")

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
    zeta = jn_zeros(0, Nr)
    k_grid = zeta / r_max
    
    
    dr = r_grid[1] - r_grid[0]
    dk = k_grid[1] - k_grid[0]

    # Precompute kernel once
    J0 = j0(np.outer(k_grid, r_grid))   # (Nk, Nr)

    # Precompute weights
    r_weight = r_grid * dr              # (Nr,)
    k_weight = k_grid * dk              # (Nk,)
    
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




























import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.special import j0, jn_zeros
import ctypes
import os
import sys
import json

from .closure import closure_update_c_matrix
from .rdf_radial import find_key_recursive

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials
# -------------------------------------------------
# Load C solver
# -------------------------------------------------
_here = os.path.dirname(__file__)

if sys.platform == "darwin":
    _libname = "liboz.dylib"
elif sys.platform == "win32":
    _libname = "liboz.dll"
else:
    _libname = "liboz.so"

_lib_path = os.path.join(_here, _libname)

if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"Shared library not found: {_lib_path}")

lib = ctypes.CDLL(_lib_path)

from ctypes import c_int
from numpy.ctypeslib import ndpointer

lib.solve_linear_system.argtypes = [
    ndpointer(dtype=c_int, flags="C_CONTIGUOUS"),
    ndpointer(dtype=c_int, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
]

# -------------------------------------------------
# Hankel transforms (2D)
# -------------------------------------------------
def hankel_transform_2d(f_r, r, k_grid, J0):
    dr = r[1] - r[0]
    return 2 * np.pi * (J0 @ (r * f_r)) * dr

def inverse_hankel_transform_2d(F_k, r, k_grid, J0):
    dk = k_grid[1] - k_grid[0]
    return (J0.T @ (k_grid * F_k)) * dk / (2*np.pi)

# -------------------------------------------------
# OZ solver (pure 2D)
# -------------------------------------------------
def solve_oz_matrix_2d(c_r, rho, r, k_grid, J0, Ns, Nr):

    Ck = np.zeros_like(c_r)
    gamma_k = np.zeros_like(c_r)

    for a in range(Ns):
        for b in range(Ns):
            Ck[a, b, :] = hankel_transform_2d(c_r[a, b, :], r, k_grid, J0)

    I = np.eye(Ns)

    for ik in range(len(k_grid)):
        C = Ck[:, :, ik]

        C_rho = C @ np.diag(rho)
        A = I - C_rho
        B = C_rho @ C

        A_f = np.asfortranarray(A)
        B_f = np.asfortranarray(B)

        lib.solve_linear_system(
            np.array([Ns], dtype=np.int32),
            np.array([Ns], dtype=np.int32),
            A_f,
            B_f
        )

        gamma_k[:, :, ik] = B_f

    gamma_r = np.zeros_like(c_r)
    for a in range(Ns):
        for b in range(Ns):
            gamma_r[a, b, :] = inverse_hankel_transform_2d(
                gamma_k[a, b, :], r, k_grid, J0
            )

    return gamma_r

# -------------------------------------------------
# MAIN PURE 2D RDF
# -------------------------------------------------
def rdf_2d(
    ctx,
    rdf_config,
    densities,
    export=True,
    plot=True,
):

    rdf_block = find_key_recursive(rdf_config, "rdf")
    species = find_key_recursive(rdf_config, "species")

    Ns = len(species)
    beta = rdf_block.get("beta", 1.0)
    n_iter = rdf_block.get("max_iteration", 200)
    tol = rdf_block.get("tolerance", 1e-5)
    filename_prefix = rdf_block.get("output_prefix", "rdf2d")
    alpha_max = rdf_block.get("alpha_max", 0.05)

    # -----------------------------
    # Grid
    # -----------------------------
    Nr = rdf_block.get("n_points", 300)
    r_max = rdf_block.get("r_max", 6.0)

    dr = r_max / (Nr + 1)
    r_grid = dr * np.arange(1, Nr + 1)

    zeta = jn_zeros(0, Nr)
    k_grid = zeta / r_max

    print("Nr =", Nr)

    # -----------------------------
    # Potentials
    # -----------------------------
    raw_data = raw_potentials(
        ctx=ctx,
        input_data=rdf_config,
        grid_points=5000,
        file_name_prefix="supplied_data_potential_raw.json",
        export_files=True
    )

    potential_dict = raw_data["potentials"]

    hc_data = hard_core_potentials(ctx=ctx, input_data=rdf_config)
    mf_data = meanfield_potentials(ctx=ctx, input_data=rdf_config)
    total_data = total_potentials(ctx=ctx, hc_source=hc_data, mf_source=mf_data)

    sigma = hc_data.get("sigma", None)
    sigma_matrix = np.zeros((Ns, Ns)) if sigma is None else np.array(sigma)

    u_matrix = np.zeros((Ns, Ns, Nr))

    for i, si in enumerate(species):
        for j in range(i, Ns):
            sj = species[j]

            pdata = potential_dict.get(si+sj) or potential_dict.get(sj+si)
            if pdata is None:
                raise KeyError(f"Missing potential for pair {si}{sj}")

            R_tab = np.asarray(pdata["r"])
            U_tab = np.asarray(pdata["U"])

            interp_u = interp1d(R_tab, U_tab, bounds_error=False, fill_value=0.0)
            u = beta * interp_u(r_grid)

            u_matrix[i, j] = u
            u_matrix[j, i] = u

    # -----------------------------
    # Closures
    # -----------------------------
    closure_cfg = rdf_block.get("closure", {})
    pair_closures = np.empty((Ns, Ns), dtype=object)

    for i, si in enumerate(species):
        for j in range(i, Ns):
            sj = species[j]

            key_ij = f"{si}{sj}"
            key_ji = f"{sj}{si}"

            closure = closure_cfg.get(key_ij) or closure_cfg.get(key_ji)
            if closure is None:
                raise KeyError(f"Missing closure for pair {key_ij}")

            pair_closures[i, j] = closure
            pair_closures[j, i] = closure

    # -----------------------------
    # Initialize
    # -----------------------------
    gamma_r = np.zeros_like(u_matrix)
    c_r = np.zeros_like(u_matrix)

    J0 = j0(np.outer(k_grid, r_grid))

    # -----------------------------
    # Iteration
    # -----------------------------
    prev_diff = np.inf
    alpha =  0.001
    for it in range(n_iter):

        # -------------------------
        # Closure update
        # -------------------------
        
        c_trial = closure_update_c_matrix(
            gamma_r, r=r_grid, pair_closures=pair_closures, u_matrix =u_matrix, sigma_matrix =  sigma_matrix
        )

   
        c_r = c_trial
        # Stabilization
        c_r = np.clip(c_r, -50.0, 50.0)

        # -------------------------
        # Solve OZ
        # -------------------------
        gamma_new = solve_oz_matrix_2d(
            c_r, densities, r_grid, k_grid, J0, Ns, Nr
        )

        # -------------------------
        # Convergence check
        # -------------------------
        delta_gamma = gamma_new - gamma_r
        diff = np.max(np.abs(delta_gamma))

        # -------------------------
        # Adaptive mixing
        # -------------------------
        gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new
        gamma_r = np.clip(gamma_r, -50.0, 50.0)

        # Adapt alpha (only occasionally to avoid noise)
        if it % 50 == 0 or diff < tol:
            if diff < prev_diff:
                alpha = min(alpha * 1.05, alpha_max)
            else:
                alpha = max(alpha * 0.5, 0.00001)

            print(f"{it:6d} | {diff:12.3e} | {alpha:8.5f}")

        if diff < tol:
            print(f"\n✅ Converged in {it+1} iterations.")
            break

        prev_diff = diff

    else:
        print(f"\n⚠️ Warning: not converged after {n_iter} iterations.")

    # -----------------------------
    # Final quantities
    # -----------------------------
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    # -----------------------------
    # Output
    # -----------------------------
    rdf_out = {}

    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            rdf_out[(si, sj)] = {
                "r": r_grid,
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
                "dimension": "2D"
            },
            "pairs": {}
        }

        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                pair_key = f"{si}{sj}"

                json_out["pairs"][pair_key] = {
                    "r": r_grid.tolist(),
                    "g_r": g_r[i, j].tolist(),
                    "h_r": (g_r[i, j] - 1.0).tolist(),
                    "c_r": c_r[i, j].tolist(),
                    "gamma_r": gamma_r[i, j].tolist(),
                    "u_r": u_matrix[i, j].tolist(),
                }

        json_path = out / f"{filename_prefix}_rdf.json"

        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=4)

        print(f"✅ RDF results exported → {json_path}")

    # -----------------------------
    # Plotting
    # -----------------------------
   

    return rdf_out
