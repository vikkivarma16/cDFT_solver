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
