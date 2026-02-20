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
from cdft_solver.generators.potential_splitter.raw import raw_potentials
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from collections.abc import Mapping
import matplotlib.pyplot as plt
import re
from cdft_solver.calculators.radial_distribution_function.closure import closure_update_c_matrix
import os
import sys
import ctypes
from ctypes import c_double, c_int, POINTER





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



'''
def solve_oz_matrix(c_r_matrix, r, densities):
    N = c_r_matrix.shape[0]

    # Forward Hankel
    c_k_matrix, k = hankel_transform_matrix_fast(c_r_matrix, r)

    # Inverse Hankel
    c_r_matrix_new = inverse_hankel_transform_matrix_fast(c_k_matrix, k, r)
    
    #c_r_matrix_new = inverse_hankel_transform_matrix_fast(c_k_matrix, k, r)

    # Check the reconstruction error
    diff_matrix = c_r_matrix_new - c_r_matrix
    total_error = np.max(np.abs(diff_matrix))
    print(f"Max absolute difference after forward+inverse Hankel: {total_error:.6e}")

    # Solve OZ in k-space
    gamma_k_matrix = np.zeros_like(c_k_matrix)
    rho_matrix = np.diag(densities)
    I = np.identity(N)
    eps_reg =1e-12
    for ik in range(len(k)):
        Ck = c_k_matrix[:, :, ik]
        num = Ck @ rho_matrix @ Ck
        A = I - Ck @ rho_matrix + eps_reg * I
        gamma_k_matrix[:, :, ik] = np.linalg.solve(A, num)

    # Inverse Hankel back to r-space
    gamma_r_matrix = inverse_hankel_transform_matrix_fast(gamma_k_matrix, k, r)

    return gamma_r_matrix
'''




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
    gamma_initial = None,
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
    if gamma_initial is not None:
        gamma_r = gamma_initial
    else:
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
    alpha = min(0.001, alpha_rdf_max)

    # -----------------------------
    # Iteration loop
    # -----------------------------
    conversion_flag  =  False
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
            conversion_flag = True
            break

        prev_diff = diff

    if (diff>tol):
        print(f"\n⚠️ Warning: not converged after {n_iter} iterations.")

    # -----------------------------
    # Final observables
    # -----------------------------
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r, conversion_flag


def apply_adaptive_ylim(ax, ydata, limit=10, clip=5):
    ymax = np.nanmax(np.abs(ydata))
    if ymax > limit:
        ax.set_ylim(-clip, clip)
def plot_matrix_quantity(
    r, quantity, u_matrix, species, title_prefix, filename, plots_dir
):
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

def rdf_alpha_r(
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
    tol = rdf_block.get("tolerence", 1e-6)
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
    
    real_potential  = raw_potentials(
        ctx=ctx,
        input_data=system,
        grid_points=5000,
        file_name_prefix="supplied_data_potential_raw.json",
        export_files=True
    )
    
    sigma = np.array ( hc_data["sigma"] )
    

    # -----------------------------
    # Build r grid
    # -----------------------------
    
    n_points = rdf_block.get("n_points", 300)
    r_max  =  rdf_block.get("r_max", 4)
    dr = r_max / (n_points + 1)
    r = dr * np.arange(1, n_points + 1)
    
    # -----------------------------
    # Closures (all ON initially)
    # -----------------------------
    closure_cfg = rdf_block.get("closure", {})
    pair_closures = np.empty((N, N), dtype=object)

    n = len(species)
    
    Nr =  n_points 
    
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
    
    potential_dict = real_potential["potentials"]
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
    # -------------------------------------------------
    # Density continuation for OZ convergence
    # -------------------------------------------------

    n_ramp = 10
    densities_target = np.asarray(densities, float)

    gamma_inputs = np.zeros_like(u_matrix)

    for step in range(1, n_ramp + 1):
        scale = step / n_ramp
        densities_step = scale * densities_target

        print(f"[OZ] Density ramp {step}/{n_ramp}  scale = {scale:.2f}")

        c_ref, gamma_ref, g_ref, conversed = multi_component_oz_solver_alpha(
            r=r,
            pair_closures=pair_closures,
            densities=densities_step,
            u_matrix=u_matrix,
            sigma_matrix=sigma_matrix,
            n_iter=n_iter,
            tol=tol,
            alpha_rdf_max=alpha_max,
            gamma_initial=gamma_inputs
        )

        if not conversed:
            raise RuntimeError(
                f"OZ solver failed to converge at density scale {scale:.2f}"
            )

        # Warm-start next step
        gamma_inputs = gamma_ref.copy()

    # Final converged solution at full density
    c_ref_final   = c_ref
    gamma_ref_final = gamma_ref
    g_ref_final   = g_ref

        
    
     
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
        Barker–Henderson radius computed by integrating
        only over the FIRST contiguous positive region
        of the integrand:
        
            integrand = 1 - exp(-beta u(r))
        """

        # --- Compute integrand
        integrand = 1.0 - np.exp(-beta * u_r)

        # --- Mask positive integrand
        positive = integrand > 0.0

        if not np.any(positive):
            # No repulsive core at all
            return 0.0, r[0]

        # --- Find first contiguous positive block
        idx = np.where(positive)[0]

        start = idx[0]
        end = start

        for k in idx[1:]:
            if k == end + 1:
                end = k
            else:
                break

        # --- Truncate to first positive region
        r_seg = r[start:end + 1]
        integrand_seg = integrand[start:end + 1]

        # --- Ensure r = 0 anchor exists (BH core condition)
        if r_seg[0] != 0.0:
            r_seg = np.insert(r_seg, 0, 0.0)
            integrand_seg = np.insert(integrand_seg, 0, 1.0)


        # --- Integrate
        d_bh = np.trapz(integrand_seg, r_seg)
        r0 = r_seg[-1]
        
        return d_bh, r0

    

    sigma_guess = np.zeros((N, N))
    has_core = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i, N):

            sigma_candidates = []

            
            g_ij = g_ref[i, j]
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

    def build_total_u_from_sigma(sigma_mat):
        u = np.zeros_like(u_matrix)
        for i in range(N):
            for j in range(i, N):
                if has_core[i, j]:
                    u[i, j] = hard_core_potential(r, sigma_mat[i, j])
                    r_m, u_m = detect_first_minimum_near_core(
                        r,
                        u_matrix[i, j],
                        sigma=sigma_mat[i, j],
                    )
                    u_att = np.zeros_like(r)
                    mask_rep = r <= r_m
                    mask_att = r > r_m
                    # WCA attractive tail
                    u_att[mask_rep] = u_m
                    u_att[mask_att] = u_matrix[i, j][mask_att]
                    u[i, j] += u_att
                else:
                    u[i, j] = u_matrix[i, j].copy()
                u[j, i] =  u[i, j]
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
        


        def unpack_sigma_vector(sigma_vec):
            sigma_mat = np.zeros((N, N))
            k = 0
            for (i, j) in hard_core_pairs:
                sigma_mat[i, j] = sigma_mat[j, i] = sigma_vec[k]
                k += 1
            return sigma_mat
            
            
            
        
        gamma_holder = {"gamma": gamma_inputs}

        def sigma_objective(sigma_vec):
            sigma_mat = unpack_sigma_vector(sigma_vec)
            u_trial = build_total_u_from_sigma(sigma_mat)
            loss = 0.0

            c_trial, gamma_trial, g_trial, conversion_flag = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities, float),
                u_matrix=u_trial,
                sigma_matrix=np.zeros((N, N)),
                n_iter=n_iter,
                tol=tol,
                alpha_rdf_max=alpha_max,
                gamma_initial=gamma_holder["gamma"]
            )

            if conversion_flag:
                gamma_holder["gamma"] = gamma_trial.copy()
            else:
                loss = 10

            for (i, j) in total_pair:
                diff = g_trial[i, j] - g_ref[i, j]
                loss += np.sum(diff * diff)

            return loss

        # -------------------------------------------------
        # Run optimizer
        # -------------------------------------------------

        sigma_init_vec = np.array([sigma_guess[i, j] for (i, j) in hard_core_pairs])
        lower_factor = 0.8
        upper_factor = 1.2

        bounds = [
            (lower_factor * s0, upper_factor * s0)
            for s0 in sigma_init_vec
        ]

        print("\nOptimizing sigma collectively across all states and pairs...")
        '''
        result = minimize(
                sigma_objective,
                sigma_init_vec,
                method="Powell",
                bounds=bounds,
                options={
                    "xtol": 1e-6,        # tighter convergence
                    "ftol": 1e-6,
                    "maxiter": 500,    # <-- MORE ITERATIONS
                    "maxfev": 20000,    # <-- MORE FUNCTION CALLS
                    "disp": True
                },
            )

        sigma_opt = unpack_sigma_vector(result.x)
        
        print("\n\n\nSigma optimized is given as:", sigma_opt , "\n\n\n")
        '''
        
        def compute_G_of_r(
            u_repulsive,
            u_attractive,
            r,
            pair_closures,
            N,
            n_alpha=20,
        ):
            """
            Computes G(r) = ∫_0^1 g_alpha(r) dα
            with debug plots of g_alpha, u_repulsive, and alpha*u_attractive
            """

            alpha_grid = np.linspace(0.0, 1.0, n_alpha)
            dalpha = alpha_grid[1] - alpha_grid[0]

           
        

            G_accum = np.zeros_like(u_attractive)
            gamma_inputs =  np.zeros_like(u_attractive)

            # --------------------------------------------------------
            # Loop over α
            # --------------------------------------------------------
            for alpha in alpha_grid:

                u_alpha = u_repulsive + alpha * u_attractive
                
                _, gamma_trial, g_alpha , conversion_flag= multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(densities, float),
                    u_matrix= u_alpha ,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tol,
                    alpha_rdf_max=alpha_max,
                    gamma_initial=gamma_inputs
                )
                
                if conversion_flag:
                    gamma_inputs =  gamma_trial.copy()
                

                # Accumulate G(r)
                G_accum += g_alpha * dalpha
            # --------------------------------------------------------
            # Compute G_u(r)
            # --------------------------------------------------------
            G_u = G_accum * u_attractive
            
            return G_u, G_accum
            
            
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

        
        u_attractive = np.zeros_like(u_matrix)
        r_minima = {}
        attractive_pairs = [ (i, j) for i in range(N) for j in range(i, N) if has_core[i, j] and np.any(u_matrix[i, j] < -1e-4) ]
        for i in range(N):
            for j in range(i, N):
                if (i, j) in attractive_pairs:
                    # Detect first minimum close to the hard core
                    r_m, u_m = detect_first_minimum_near_core(
                        r,
                        u_matrix[i, j],
                        sigma=sigma[i, j],
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
                    print ("\n\n\n\n\n\n\n", r_m, "\n\n\n\n\n\n")
                    print ("\n\n\n\n\n\n\n", u_m, "\n\n\n\n\n\n")

        # ============================================================
        # Run G(r) computation for σ_opt
        # ============================================================
        G_u, G_accume = compute_G_of_r(
            u_repulsive = u_ref,
            u_attractive = u_attractive,
            r=r,
            pair_closures=pair_closures,
            N=N,
            n_alpha=20
        )
        
    G_out = {}
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            G_out[(si, sj)] = {
                "r": r,
                "g_r": G_accume[i, j],
                "g_u_r": G_u[i, j],
            }
    
    return G_out

