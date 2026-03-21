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
from cdft_solver.generators.potential_splitter.raw import raw_potentials 
from cdft_solver.calculators.radial_distribution_function.closure import closure_update_c_matrix
from scipy.interpolate import interp1d
import os
import sys
import ctypes
from ctypes import c_double, c_int, POINTER


hard_core_repulsion = 1e6

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
    alpha = min(0.01, alpha_rdf_max)

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







    
    
    
def c_analysis(
    ctx,
    rdf_config,
    supplied_data,
    densities,
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
    
    
    raw_potential = raw_potentials(
        ctx=ctx,
        input_data=system,
        grid_points=5000,
        file_name_prefix="supplied_data_potential_raw.json",
        export_files=True
    )
    
    pdata = raw_potential["potentials"]
    
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
    potential_dict = pdata
    u_matrix = np.zeros((N, N, len(r)))
    print ("closures applied: ", pair_closures)
    N = len (species)
    n = len (species)
    
    
    print ("reference beta: ", beta_ref)
    # -------------------------------------------------
    # Initialize σ and u
    # -------------------------------------------------
    sigma_matrix = np.zeros((N, N)) if sigma is None else sigma.copy()
    
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
                fill_value=(pdata["U"][0], 0.0),
                assume_sorted=True,
            )
            u_val = beta_ref * interp_u(r)
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
    
    
    densities_s = densities
    beta_s = beta
    c_pred, gamma_pred, g_pred, conversion_flag = multi_component_oz_solver_alpha(
        r=r,
        pair_closures=pair_closures,
        densities=np.asarray(densities_s, float),
        u_matrix=beta_s * u_matrix / beta_ref,
        sigma_matrix=sigma_matrix,
        n_iter=n_iter,
        tol=tolerance,
        alpha_rdf_max=alpha_max,
    )

    final_oz_results = {
        "beta": beta_s,
        "densities": np.asarray(densities_s, float),
        "g_pred": g_pred.copy(),
        "c_pred": c_pred.copy(),
        "gamma_pred": gamma_pred.copy(),
    }
    
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
    
    sigma_guess = np.zeros((N, N))
    has_core = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i, N):

            g_ij = final_oz_results["g_pred"][i, j]
            sigma_s = detect_sigma_from_gr(r, g_ij)

            if sigma_s > 0.0:
                sigma_guess[i, j] = sigma_guess[j, i] = sigma_s
                has_core[i, j] = has_core[j, i] = True

                print(f"Detected hard core for pair ({i},{j}) : σ ≈ {sigma_s:.4f}")


    hard_core_pairs = [(i, j) for i in range(N) for j in range(i, N) if has_core[i, j]]
    
    
    
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

                    u_att[mask_rep] = u_m
                    u_att[mask_att] = u_matrix[i, j][mask_att]

                    u[i, j] += u_att

                else:

                    u[i, j] = u_matrix[i, j].copy()

                u[j, i] = u[i, j]
        return u
        
        

    total_pair = [(i, j) for i in range(N) for j in range(i, N)]
    if hard_core_pairs:

        print("\n🔧 Starting sigma calibration stage...")


        def unpack_sigma_vector(sigma_vec):

            sigma_mat = np.zeros((N, N))
            k = 0

            for (i, j) in hard_core_pairs:

                sigma_mat[i, j] = sigma_mat[j, i] = sigma_vec[k]
                k += 1

            return sigma_mat


        def sigma_objective(sigma_vec):

            sigma_mat = unpack_sigma_vector(sigma_vec)

            u_trial = build_total_u_from_sigma(sigma_mat)

            c_trial, gamma_trial, g_trial, conversion_flag = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities_s, float),
                u_matrix=beta_s * u_trial / beta_ref,
                sigma_matrix=np.zeros((N, N)),
                n_iter=n_iter,
                tol=tolerance,
                alpha_rdf_max=alpha_max,
            )

            if not conversion_flag:
                return 10

            loss = 0.0

            for (i, j) in total_pair:

                diff = g_trial[i, j] - final_oz_results["g_pred"][i, j]

                loss += np.sum(diff * diff)

            return loss


        sigma_init_vec = np.array([sigma_guess[i, j] for (i, j) in hard_core_pairs])

        bounds = [(0.9 * s0, 1.2 * s0) for s0 in sigma_init_vec]

        result = minimize(
            sigma_objective,
            sigma_init_vec,
            method="Powell",
            bounds=bounds,
            options={"xtol": 1e-6, "ftol": 1e-6, "maxiter": 500, "disp": True},
        )

        sigma_opt = unpack_sigma_vector(result.x)


        print("Sigma reference system analysis\n")


        u_ref = np.zeros_like(u_matrix)

        for i in range(N):
            for j in range(N):

                if has_core[i, j]:
                    u_ref[i, j] = wca_split(r, u_matrix[i, j])
                else:
                    u_ref[i, j] = u_matrix[i, j].copy()


        c_ref, gamma_ref, g_ref, conversion_flag = multi_component_oz_solver_alpha(
            r=r,
            pair_closures=pair_closures,
            densities=np.asarray(densities_s, float),
            u_matrix=u_ref,
            sigma_matrix=np.zeros((N, N)),
            n_iter=n_iter,
            tol=tolerance,
            alpha_rdf_max=alpha_max,
        )


        bh_zero = {}
        bh_sigma = np.zeros_like(sigma_opt)

        for (i, j) in hard_core_pairs:

            d_bh, r0 = compute_bh_radius_truncated(r, u_matrix[i, j], beta_ref)

            bh_zero[(i, j)] = r0
            bh_sigma[i, j] = bh_sigma[j, i] = d_bh


        def compute_repulsive_gr(sigma_mat):

            u_rep = build_total_u_from_sigma(sigma_mat)

            c_state, gamma_state, g_state, conversion_flag = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities_s, float),
                u_matrix=beta_s * u_rep / beta_ref,
                sigma_matrix=np.zeros((N, N)),
                n_iter=n_iter,
                tol=tolerance,
                alpha_rdf_max=alpha_max,
            )

            return g_state, c_state, gamma_state


        g_rep_sigma_opt, c_rep_sigma_opt, gamma_rep_sigma_opt = compute_repulsive_gr(sigma_opt)

        g_rep_sigma_bh, c_rep_sigma_bh, gamma_rep_sigma_bh = compute_repulsive_gr(bh_sigma)


        u_rep = build_total_u_from_sigma(sigma_opt)


        reference_package = {

            "sigma_opt": sigma_opt.tolist(),
            "sigma_bh": bh_sigma.tolist(),

            "bh_meta": {
                f"{i},{j}": {"d_bh": bh_sigma[i, j], "r0": bh_zero[(i, j)]}
                for (i, j) in hard_core_pairs
            },

            "r": r.tolist(),
            "u_ref": u_ref.tolist(),
            "u_real": u_matrix.tolist(),
            "u_sigma_opt": u_rep.tolist(),

            "g_ref_hard": g_ref.tolist(),
            "c_ref_hard": c_ref.tolist(),
            "gamma_ref_hard": gamma_ref.tolist(),

            "g_rep_sigma_opt": g_rep_sigma_opt.tolist(),
            "c_rep_sigma_opt": c_rep_sigma_opt.tolist(),
            "gamma_rep_sigma_opt": gamma_rep_sigma_opt.tolist(),

            "g_rep_sigma_bh": g_rep_sigma_bh.tolist(),
            "c_rep_sigma_bh": c_rep_sigma_bh.tolist(),
            "gamma_rep_sigma_bh": gamma_rep_sigma_bh.tolist(),

            "g_real": final_oz_results["g_pred"].tolist(),
            "c_real": final_oz_results["c_pred"].tolist(),
            "gamma_real": final_oz_results["gamma_pred"].tolist(),
        }


        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_file = out / "result_sigma_analysis.json"

        with open(json_file, "w") as f:
            json.dump(reference_package, f, indent=4)

        print("✅ Saved result_sigma_analysis.json")
                
        
        
        
        
        
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
                        sigma=bh_sigma[i, j],
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
        
        
        
        
        
        # ============================================================
        # G(r) computation (state-free)
        # ============================================================
        
        
        

        def compute_G_of_r(
            u_repulsive,
            u_attractive,
            densities,
            beta,
            r,
            pair_closures,
            beta_ref,
            N,
            n_iter,
            tolerance,
            alpha_max,
            n_alpha=20,
        ):
            """
            Computes

                G(r) = ∫_0^1 g_alpha(r) dα

            for a single thermodynamic state.
            """

            alpha_grid = np.linspace(0.0, 1.0, n_alpha)
            dalpha = alpha_grid[1] - alpha_grid[0]

            G_accum = np.zeros_like(u_attractive)

            for alpha in alpha_grid:

                u_alpha = u_repulsive + alpha * u_attractive

                _, _, g_alpha, conversion_flag = multi_component_oz_solver_alpha(
                    r=r,
                    pair_closures=pair_closures,
                    densities=np.asarray(densities, float),
                    u_matrix= u_alpha,
                    sigma_matrix=np.zeros((N, N)),
                    n_iter=n_iter,
                    tol=tolerance,
                    alpha_rdf_max=alpha_max,
                )

                if not conversion_flag:
                    raise RuntimeError("OZ solver failed during alpha integration")

                G_accum += g_alpha * dalpha

            G_u = beta * G_accum * u_attractive
            
            
            return G_accum, G_u


        # ============================================================
        # Run G(r) computation
        # ============================================================

        G_r_sigma_opt, G_u_r_sigma_opt = compute_G_of_r(
            u_repulsive=build_hard_core_u_from_sigma(sigma_opt),
            u_attractive=u_attractive,
            densities=densities,
            beta=beta_ref,
            r=r,
            pair_closures=pair_closures,
            beta_ref=beta_ref,
            N=N,
            n_iter=n_iter,
            tolerance=tolerance,
            alpha_max=alpha_max,
        )

        G_r_real, G_u_r_real = compute_G_of_r(
            u_repulsive=u_ref,
            u_attractive=u_attractive,
            densities=densities,
            beta=beta_ref,
            r=r,
            pair_closures=pair_closures,
            beta_ref=beta_ref,
            N=N,
            n_iter=n_iter,
            tolerance=tolerance,
            alpha_max=alpha_max,
        )
        
        
        
        # ============================================================
        # Second virial coefficient analysis (ΔB2)
        # ============================================================

        def compute_B2(r, u_r, beta):
            """
            Compute second virial coefficient:
                B2 = -2π ∫ (exp(-β u(r)) - 1) r^2 dr
            """
            f_r = np.exp(-beta * u_r) - 1.0
            return -2.0 * np.pi * np.trapz(r**2 * f_r, r)


        B2_real = np.zeros((N, N))
        B2_ref  = np.zeros((N, N))
        delta_B2 = np.zeros((N, N))
        two_delta_B2 = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):

                # --- Full system
                B2_real[i, j] = compute_B2(r, u_matrix[i, j], beta_ref)

                # --- Reference system
                B2_ref[i, j]  = compute_B2(r, u_ref[i, j], beta_ref)

                # --- ΔB2
                delta = B2_real[i, j] - B2_ref[i, j]

                delta_B2[i, j] = delta_B2[j, i] = delta
                two_delta_B2[i, j] = two_delta_B2[j, i] = 2.0 * delta
                
                
        delta_b2_package = {
            "B2_real": B2_real.tolist(),
            "B2_ref": B2_ref.tolist(),
            "delta_B2": delta_B2.tolist(),
            "2_delta_B2": two_delta_B2.tolist(),
        }

        out_file = Path(ctx.scratch_dir) / "delta_B2_results.json"

        with open(out_file, "w") as f:
            json.dump(delta_b2_package, f, indent=4)

        print("✅ ΔB2 analysis exported →", out_file)


        # ============================================================
        # Export results
        # ============================================================

        attractive_package_g = {
            "sigma_opt": sigma_opt.tolist(),
            "r": r.tolist(),
            "G_r_sigma_opt": G_r_sigma_opt.tolist(),
            "G_r_real": G_r_real.tolist(),
            "G_u_r_sigma_opt": G_u_r_sigma_opt.tolist(),
            "G_u_r_real": G_u_r_real.tolist(),
            "u_attractive_real": u_attractive.tolist()
        }

        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_file = out / "result_G_of_r.json"

        with open(json_file, "w") as f:
            json.dump(attractive_package_g, f, indent=4)

        print("✅ G(r) and u_attractive exported →", json_file)


        # ============================================================
        # Repulsive RDF using sigma
        # ============================================================

        def compute_repulsive_gr_hard(sigma_mat):

            u_rep = build_hard_core_u_from_sigma(sigma_mat)

            c_state, gamma_state, g_state, conversion_flag = multi_component_oz_solver_alpha(
                r=r,
                pair_closures=pair_closures,
                densities=np.asarray(densities, float),
                u_matrix=beta_ref * u_rep / beta_ref,
                sigma_matrix=np.zeros((N, N)),
                n_iter=n_iter,
                tol=tolerance,
                alpha_rdf_max=alpha_max,
            )

            if not conversion_flag:
                raise RuntimeError("OZ solver failed")

            return g_state, c_state, gamma_state


        g_hard_sigma_opt, c_hard_sigma_opt, gamma_hard_sigma_opt = compute_repulsive_gr_hard(sigma_opt)


        # ============================================================
        # Collect c(r)
        # ============================================================

        c_real = np.asarray(final_oz_results["c_pred"])
        c_ref_hard = np.asarray(c_ref)
        c_sigma_opt = np.asarray(reference_package["c_rep_sigma_opt"])
        c_rep_sigma_opt = np.asarray(c_hard_sigma_opt)


        # ============================================================
        # Compute Δc(r)
        # ============================================================

        delta_c_real_ref = -(c_real - c_ref_hard)
        delta_c_real_sigma_opt = -(c_real - c_rep_sigma_opt)
        delta_c_sigma_opt_ref = -(c_sigma_opt - c_ref_hard)
        delta_c_sigma_opt_sigma_opt = -(c_sigma_opt - c_rep_sigma_opt)


        # ============================================================
        # Export Δc package
        # ============================================================

        delta_c_package = {
            "r": r.tolist(),
            "c_real": c_real.tolist(),
            "c_ref_hard": c_ref_hard.tolist(),
            "c_sigma_opt": c_sigma_opt.tolist(),
            "c_rep_sigma_opt": c_rep_sigma_opt.tolist(),
            "delta_c_real_ref": delta_c_real_ref.tolist(),
            "delta_c_real_sigma_opt": delta_c_real_sigma_opt.tolist(),
            "delta_c_sigma_opt_ref": delta_c_sigma_opt_ref.tolist(),
            "delta_c_sigma_opt_sigma_opt": delta_c_sigma_opt_sigma_opt.tolist(),
        }

        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        out_file = out / "delta_c_results.json"

        with open(out_file, "w") as f:
            json.dump(delta_c_package, f, indent=4)

        print("✅ c(r) + Δc(r) exported →", out_file)


        # ============================================================
        # Potential export
        # ============================================================

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

            print(f"✅ Inverted potential exported: {json_file}")


        # ============================================================
        # Potential plots
        # ============================================================

        if export_plot:

            plots_dir = getattr(ctx, "plots_dir", ctx.scratch_dir)
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)

            for i, si in enumerate(species):
                for j, sj in enumerate(species):

                    plt.figure(figsize=(6,4))

                    plt.plot(r, u_matrix[i, j] / beta_ref)

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
                    plt.tight_layout()

                    plt.ylim(-10,10)

                    plt.savefig(
                        plots_dir / f"{filename_prefix}_potential_{si}{sj}.png",
                        dpi=600
                    )

                    plt.close()

            print(f"✅ Potential plots exported → {plots_dir}")


        return u_matrix / beta_ref, sigma_matrix
