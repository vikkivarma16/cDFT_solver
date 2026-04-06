import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.special import j0
import ctypes
import os
import sys

from .closure import closure_update_c_matrix
from .rdf_radial import find_key_recursive

from cdft_solver.generators.grids_properties.k_and_r_space_cylindrical import r_k_space_cylindrical
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.total import total_potentials

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

    # Forward Hankel
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

    # Inverse Hankel
    gamma_r = np.zeros_like(c_r)
    for a in range(Ns):
        for b in range(Ns):
            gamma_r[a, b, :] = inverse_hankel_transform_2d(
                gamma_k[a, b, :], r, k_grid, J0
            )

    return gamma_r

# -------------------------------------------------
# Plot
# -------------------------------------------------
def plot_rdf(g_r, r_grid, Ns, ctx):
    plots = Path(ctx.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(Ns, Ns, figsize=(4*Ns, 4*Ns))

    if Ns == 1:
        axes = np.array([[axes]])

    for i in range(Ns):
        for j in range(Ns):
            ax = axes[i, j]
            ax.plot(r_grid, g_r[i, j])
            ax.set_title(f"g_{i}{j}(r)")
            ax.grid(True)

    plt.tight_layout()
    fname = plots / "rdf_2d.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"Saved {fname}")

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
    n_iter = find_key_recursive(rdf_config, "max_iteration")
    tol = rdf_block.get("tolerance", 1e-5)

    # -----------------------------
    # Grid (ONLY radial now)
    # -----------------------------
    
    n_points = rdf_block.get("n_points", 300)
    Nr = n_points
    r_max  =  rdf_block.get("r_max", 6)
    dr = r_max / (n_points + 1)
    r_grid = dr * np.arange(1, n_points + 1)
        
    alpha = jn_zeros(0, Nr)   # zeros of J0
    k_grid = alpha / r_max
    
    

    Nr = len(r_grid)

    print("Nr =", Nr)

    # -----------------------------
    # Potentials (pure radial)
    # -----------------------------
    hc_data = hard_core_potentials(ctx=ctx, input_data=rdf_config)
    mf_data = meanfield_potentials(ctx=ctx, input_data=rdf_config)
    total_data = total_potentials(ctx=ctx, hc_source=hc_data, mf_source=mf_data)

    potential_dict = total_data["total_potentials"]

    u_matrix = np.zeros((Ns, Ns, Nr))

    for i, si in enumerate(species):
        for j in range(i, Ns):

            sj = species[j]
            pdata = potential_dict.get(si+sj) or potential_dict.get(sj+si)

            R_tab = np.asarray(pdata["r"])
            U_tab = np.asarray(pdata["U"])

            interp_u = interp1d(R_tab, U_tab, bounds_error=False, fill_value=0.0)

            u = beta * interp_u(r_grid)

            u_matrix[i, j] = u
            u_matrix[j, i] = u

    # -----------------------------
    # Initialize
    # -----------------------------
    gamma_r = np.zeros_like(u_matrix)
    c_r = np.zeros_like(u_matrix)

    J0 = j0(np.outer(k_grid, r_grid))

    # -----------------------------
    # Iteration
    # -----------------------------
    alpha = 0.1

    for it in range(n_iter):

        gamma_old = gamma_r.copy()

        c_trial = closure_update_c_matrix(
            gamma_r.reshape(Ns, Ns, Nr),
            r_grid,
            None,
            u_matrix.reshape(Ns, Ns, Nr),
            None
        )

        c_r[:] = c_trial

        gamma_new = solve_oz_matrix_2d(
            c_r, densities, r_grid, k_grid, J0, Ns, Nr
        )

        gamma_r = (1 - alpha) * gamma_old + alpha * gamma_new

        err = np.max(np.abs(gamma_r - gamma_old))

        if it % 10 == 0:
            print(f"Iter {it} | err = {err:.3e}")

        if err < tol:
            print("✅ Converged")
            break

    h_r = gamma_r + c_r
    g_r = h_r + 1.0
    
    
    
    
    
    
    
    
    
    
    # ============================================================
    # Output
    # ============================================================
    import json

    r = r_grid  # unify naming

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
                "dimension": "2D"
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

        # filename-friendly density string
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
