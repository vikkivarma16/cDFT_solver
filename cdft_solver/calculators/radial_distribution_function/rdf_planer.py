import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.special import j0

from .closure import closure_update_c_matrix
from .rdf_radial import find_key_recursive
import ctypes

from ctypes import CDLL, c_int
from numpy.ctypeslib import ndpointer
import os
import sys

from cdft_solver.generators.grids_properties.k_and_r_space_cylindrical import r_k_space_cylindrical
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
# -------------------------------------------------
# Hankel DST transforms (radial)
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

# Load shared library
lib = ctypes.CDLL(_lib_path)



lib.solve_linear_system.argtypes = [
    ndpointer(dtype=c_int, flags="C_CONTIGUOUS"),
    ndpointer(dtype=c_int, flags="C_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
]







from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_functions(gamma_r, c_r, r_grid, z_grid, Ns, Nz, Nr, ctx, plot=True):
    """
    Plot and save RDF results.
    - Radial correlations at diagonal z_i = z_j
    - Plane–plane correlations at r = 0
    """
    if not plot:
        return
    
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    # Create plots directory
    plots = Path(ctx.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1️⃣ Radial correlations (r vs h(r) for z_i=z_j)
    # -----------------------------
    curves_per_fig = 5
    n_figs = int(np.ceil(Nz / curves_per_fig))
    cmap = plt.cm.RdBu
    norm = plt.Normalize(vmin=0, vmax=Nz-1)

    for fig_id in range(n_figs):
        i_start = fig_id * curves_per_fig
        i_end = min((fig_id+1) * curves_per_fig, Nz)
        i_debug = np.arange(i_start, i_end)

        fig, axes = plt.subplots(Ns, Ns, figsize=(4*Ns+1.2, 4*Ns), sharex=True, sharey=True)
        if Ns == 1:
            axes = np.array([[axes]])

        for a in range(Ns):
            for b in range(Ns):
                ax = axes[a, b]
                for i in i_debug:
                    color = cmap(norm(i))
                    hr_vals = h_r[a, b, i, i, :]
                    ax.plot(r_grid, hr_vals, color=color, alpha=0.9, linewidth=1.5)
                ax.set_title(rf"$h_{{{a}{b}}}(z_i=z_j,r)$")
                ax.grid(True)
                if a == Ns-1:
                    ax.set_xlabel(r"$r$")
                if b == 0:
                    ax.set_ylabel(r"$h(r)$")

        plt.suptitle(f"Radial correlations (z_i=z_j) — planes {i_start} to {i_end-1}")
        plt.tight_layout()
        fname = plots / f"h_r_diag_z_planes_{i_start:03d}_{i_end-1:03d}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")

    # -----------------------------
    # 2️⃣ Plane–plane correlations at r=0
    # -----------------------------
    r0_idx = 0  # index for r=0
    for fig_id in range(n_figs):
        i_start = fig_id * curves_per_fig
        i_end = min((fig_id+1) * curves_per_fig, Nz)
        i_debug = np.arange(i_start, i_end)

        fig, axes = plt.subplots(Ns, Ns, figsize=(4*Ns+1.2, 4*Ns), sharex=True, sharey=True)
        if Ns == 1:
            axes = np.array([[axes]])

        for a in range(Ns):
            for b in range(Ns):
                ax = axes[a,b]
                for i in i_debug:
                    color = cmap(norm(i))
                    # + side (z_j > z_i)
                    if i < Nz-1:
                        dz_plus = z_grid[i+1:] - z_grid[i]
                        h_plus = h_r[a,b,i,i+1:,r0_idx]
                        ax.plot(dz_plus, h_plus, color=color, alpha=0.9, linewidth=1.4)
                    # - side (z_j < z_i)
                    if i > 0:
                        dz_minus = z_grid[i] - z_grid[:i]
                        h_minus = h_r[a,b,i,:i,r0_idx][::-1]
                        ax.plot(dz_minus, h_minus, color=color, alpha=0.6, linewidth=1.4, linestyle='--')
                    # origin point
                    ax.plot(0, h_r[a,b,i,i,r0_idx], 'ko')

                ax.set_title(rf"$h_{{{a}{b}}}(z_i,z_j,r=0)$")
                ax.grid(True)
                if a == Ns-1:
                    ax.set_xlabel(r"$|z_j - z_i|$")
                if b == 0:
                    ax.set_ylabel(r"$h(z_i,z_j,0)$")

        plt.suptitle(f"Plane–plane correlations at r=0 — planes {i_start} to {i_end-1}")
        plt.tight_layout()
        fname = plots / f"h_z_r0_planes_{i_start:03d}_{i_end-1:03d}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")








'''

def solve_oz_realspace_planar(h_r, densities, r_grid, z_grid):
    """
    Solve the planar OZ equation in real space:
        c = h * (I + rho * h)^(-1)
    for multicomponent, z-dependent systems.

    Parameters
    ----------
    h_r : ndarray
        Total correlation function, shape (Ns, Ns, Nz, Nz, Nr)
        h_r[a,b,i,j,r] corresponds to species a,b, planes z_i,z_j, and radial point r.
    densities : array_like
        Species densities, shape (Ns,)
    r_grid : ndarray
        Radial grid, shape (Nr,)
    z_grid : ndarray
        Planar z-grid, shape (Nz,)

    Returns
    -------
    c_r : ndarray
        Direct correlation function, same shape as h_r
    """
    Ns, _, Nz, _, Nr = h_r.shape
    Nd = Ns * Nz  # total number of species*planes
    c_r = np.zeros_like(h_r)

    # Build rho_diag: diagonal matrix in ((species, z)) space
    rho_diag = np.repeat(densities, Nz)
    rho_diag_matrix = np.diag(rho_diag)

    for ir in range(Nr):
        # Flatten h_r at this r: (Ns*Nz, Ns*Nz)
        H_flat = h_r[..., ir].reshape(Ns, Ns, Nz, Nz).transpose(0,2,1,3).reshape(Nd, Nd)

        # Compute c using OZ inversion
        # C = H @ (I + rho * H)^(-1)
        M = np.eye(Nd) + rho_diag_matrix @ H_flat
        try:
            C_flat = H_flat @ np.linalg.inv(M)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"OZ inversion failed at r index {ir}, matrix may be ill-conditioned.")

        # Reshape back to (Ns, Ns, Nz, Nz)
        C_reshaped = C_flat.reshape(Ns, Nz, Ns, Nz).transpose(0,2,1,3)
        c_r[..., ir] = C_reshaped

    return c_r

'''

def hankel_transform_2d(f_r, r, k_grid, J0):
    """Apply Hankel transform for cylindrical coordinates along r."""
    dr = r[1] - r[0]
    return 2 * np.pi * (J0 @ (r * f_r)) * dr

def inverse_hankel_transform_2d(F_k, r, k_grid, J0):
    dk = k_grid[1] - k_grid[0]
    return (J0.T @ (k_grid * F_k)) * dk / (2*np.pi)
    


# -------------------------------------------------
# Solve OZ in k-space
# -------------------------------------------------


def solve_oz_matrix_2d(c_r, RHO, r_grid, k_grid, J0, Ns, Nz, Nr):
    # Flatten i,j
    Nd = Ns * Nz
    c_flat = c_r.reshape(Ns, Ns, Nz*Nz, Nr)
    gamma_flat = np.zeros_like(c_flat)

    # Vectorized Hankel transform
    r_f = c_flat * r_grid[None,None,None,:]  # shape: Ns,Ns,Nz*Nz,Nr
    Ck_flat = 2*np.pi * np.tensordot(r_f, J0.T, axes=([3],[0])) * (r_grid[1]-r_grid[0])  # Ns,Ns,Nz*Nz,len(k)

    # OZ solve using C solver per k
    for ik in range(len(k_grid)):
        C_big = Ck_flat[..., ik].reshape(Nd, Nd)
        C_rho = C_big @ RHO
        A = np.eye(Nd) - C_rho
        B = C_rho @ C_big
        A_f = np.asfortranarray(A)
        B_f = np.asfortranarray(B)
        lib.solve_linear_system(np.array([Nd],dtype=np.int32), np.array([Nd],dtype=np.int32), A_f, B_f)
        gamma_flat[..., ik] = B_f.reshape(Ns,Ns,Nz*Nz)

    # Inverse Hankel
    r_f = gamma_flat * k_grid[None,None,None,:]
    gamma_r_flat = np.tensordot(r_f, J0, axes=([3],[0])) * (k_grid[1]-k_grid[0]) / (2*np.pi)
    gamma_r = gamma_r_flat.reshape(Ns, Ns, Nz, Nz, Nr)  # <-- reshape to original 5D
    return gamma_r



def solve_oz_matrix_2d(c_r_matrix, RHO, r, k_grid, J0, Ns, Nz, Nr):
    """
    2D OZ solver along r for each plane z_i, z_j using the C solver.
    """
    Ck = np.zeros_like(c_r_matrix)
    gamma_k = np.zeros_like(c_r_matrix)

    # Forward Hankel along r
    for a in range(Ns):
        for b in range(Ns):
            for i in range(Nz):
                for j in range(Nz):
                    Ck[a, b, i, j, :] = hankel_transform_2d(c_r_matrix[a, b, i, j, :], r, k_grid, J0)

    Nd = Ns * Nz
    I_Nd = np.eye(Nd)

    # Pre-allocate matrices for C solver
    A = np.empty((Nd, Nd))
    B = np.empty((Nd, Nd))

    N_ptr = np.array([Nd], dtype=np.int32)
    M_ptr = np.array([Nd], dtype=np.int32)

    for ik in range(len(k_grid)):
        # Flatten to (Nd, Nd) per k
        C_big = Ck[..., ik].transpose(0, 2, 1, 3).reshape(Nd, Nd)

        # Construct matrices
        C_rho = C_big @ RHO
        A[:] = I_Nd - C_rho
        B[:] = C_rho @ C_big

        # Fortran-contiguous arrays for C solver
        A_f = np.asfortranarray(A)
        B_f = np.asfortranarray(B)

        # Call C solver
        lib.solve_linear_system(N_ptr, M_ptr, A_f, B_f)

        # Reshape back to (Ns, Nz, Ns, Nz)
        gamma_k[..., ik] = B_f.reshape(Ns, Nz, Ns, Nz).transpose(0, 2, 1, 3)

    # Inverse Hankel
    gamma_r = np.zeros_like(c_r_matrix)
    for a in range(Ns):
        for b in range(Ns):
            for i in range(Nz):
                for j in range(Nz):
                    gamma_r[a, b, i, j, :] = inverse_hankel_transform_2d(gamma_k[a, b, i, j, :], r, k_grid, J0)

    return gamma_r




# -------------------------------------------------
# Main 2D RDF solver
# -------------------------------------------------
def rdf_planer(
    ctx,
    rdf_config,
    densities,
    supplied_data=None,
    export=True,
    plot=True,
    filename_prefix="rdf_2d"
):

    # -----------------------------
    # Extract RDF parameters
    # -----------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    if rdf_block is None:
        raise KeyError("No 'rdf' key found in rdf_config")

    species = find_key_recursive(rdf_config, "species")
    Ns = len(species)

    beta = rdf_block.get("beta", 1.0)
    tol = rdf_block.get("tolerance", 1e-6)
    n_iter = find_key_recursive(rdf_config, "max_iteration")
    alpha_max = rdf_block.get("alpha_max", 0.05)
    
    
    rdf_planer  =  find_key_recursive(rdf_config, "planer_rdf")
    planer_grid_config = {}
    planer_grid_config ["space_confinement_parameters"] = rdf_planer
    
    r_k_grid_planer = r_k_space_cylindrical(ctx = ctx,  data_dict =  planer_grid_config, export_json = True, filename = "supplied_data_r_k_space_box_planer.json")
    
    
    r_space =  np.array(r_k_grid_planer["r_space"])
    k_space =  np.array(r_k_grid_planer["k_space"])
    # k-grid (radial k)
    k_grid = np.sort(np.unique(k_space[:, 1]))
    
    
    #print ("old grid k", k_grid)
    
    z_grid = np.sort(np.unique(r_space[:, 0]))
    r_grid = np.sort(np.unique(r_space[:, 1]))
    
    Nr = len(r_grid)
    Nz = len(z_grid)
    
    #k_grid = np.linspace(0.0, 20.0, Nr)
    
    #print(k_grid)

    print ("supplied radial grid points are :", Nr)
    print ("supplied planer points are :", Nz)

    # -----------------------------
    # Build potentials (same strategy as rdf_radial)
    # -----------------------------
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
        hc_source=hc_data,
        mf_source=mf_data,
        file_name_prefix="supplied_data_potential_total.json",
        export_files=False,
    )

    potential_dict = total_data["total_potentials"]
    sigma = hc_data["sigma"]



    closure_cfg = find_key_recursive(rdf_config, "closure")
    if closure_cfg is None:
        raise KeyError("No closure definitions found")

    pair_closures = np.empty((Ns, Ns), dtype=object)

    for i, si in enumerate(species):
        for j in range(i, Ns):
            sj = species[j]

            key_ij = si + sj
            key_ji = sj + si

            if key_ij in closure_cfg:
                closure = closure_cfg[key_ij]
            elif key_ji in closure_cfg:
                closure = closure_cfg[key_ji]
            else:
                raise KeyError(
                    f"Missing closure for pair '{key_ij}' or '{key_ji}'"
                )

            pair_closures[i, j] = closure
            pair_closures[j, i] = closure
            
    # -----------------------------
    # Sigma matrix
    # -----------------------------
    
    print ("supplied pair closures are given as:", pair_closures)
    
    sigma_matrix = np.zeros((Ns,Ns)) if sigma is None else np.array(sigma)
    
    print ("Given sigma matrix is :", sigma_matrix)
    # -----------------------------
    # Initialize correlation functions
    # -----------------------------
    Zij = z_grid[:, None] - z_grid[None, :]
    dz = z_grid[1] - z_grid[0]
    
    
    #print(z_grid)
    
    # Geometry
    R_ijr = np.sqrt(Zij[:, :, None]**2 + r_grid[None, None, :]**2)
    
    #print ("\n\n\n length one :", len(R_ijr))
    #print ("\n\n\n length two :", len(R_ijr[0]))
    #print ("\n\n\n length three :", len(R_ijr[0][0]))
    
    
    u_matrix = np.zeros((Ns, Ns, Nz, Nz, Nr))

    for i, si in enumerate(species):
        for j in range(i, Ns):
            sj = species[j]

            key_ij = si + sj
            key_ji = sj + si

            pdata = potential_dict.get(key_ij) or potential_dict.get(key_ji)
            if pdata is None:
                raise KeyError(f"Missing potential for pair '{si}-{sj}'")

            R_tab = np.asarray(pdata["r"], dtype=float)
            U_tab = np.asarray(pdata["U"], dtype=float)

            R_cut = R_tab.max()

            interp_u = interp1d(
                R_tab,
                U_tab,
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )

            # Vectorized potential evaluation
            u_val = beta * interp_u(R_ijr)

            # Explicit cutoff enforcement (optional but robust)
            u_val[R_ijr > R_cut] = 0.0
 
            # Symmetric assignment
            u_matrix[i, j] = u_val
            u_matrix[j, i] = u_val

        
    
    gamma_r = np.zeros_like(u_matrix)
    c_r = np.zeros_like(u_matrix)

    # -----------------------------
    # Supplied data processing
    # -----------------------------
    
    # -----------------------------
    # Main OZ iteration with dynamic alpha
    # -----------------------------
    alpha = 0.1       # initial mixing fraction
    alpha_min = 0.01
    relax_increase = 1.05
    relax_decrease = 0.7
    
    Nd = Ns * Nz
    rho_flat = densities.reshape(Nd)
    RHO = np.diag(rho_flat * dz)
    
    J0 = j0(np.outer(k_grid, r_grid))

    for it in range(n_iter):
        gamma_old = gamma_r.copy()

        # (1) Update c from closure
        c_trial = closure_update_c_matrix(
            gamma_r.reshape(Ns,Ns,Nz*Nz*Nr),
            R_ijr.reshape(Nz*Nz*Nr),
            pair_closures,
            u_matrix.reshape(Ns,Ns,Nz*Nz*Nr),
            sigma_matrix
        )
        c_r[:] = c_trial.reshape(Ns,Ns,Nz,Nz,Nr)
        
        

        # (2) Solve OZ
        gamma_new = solve_oz_matrix_2d(c_r, RHO, r_grid, k_grid, J0, Ns, Nz, Nr)
        

        # (3) Dynamic alpha mixing
        gamma_r = (1 - alpha) * gamma_old + alpha * gamma_new

        # (4) Convergence check
        err = np.max(np.abs(gamma_r - gamma_old))

        # adapt alpha
        
        if err > 2 * tol:
            # possibly oscillating, reduce alpha
            alpha = max(alpha * relax_decrease, alpha_min)
        else:
            # stable convergence, increase alpha gradually
            alpha = min(alpha * relax_increase, alpha_max)
        
        if it % 10 == 0:
            print(f"Iteration {it} | Δγ = {err:.3e} | α = {alpha:.3f}")

        if err < tol:
            print(f"✅ Converged after {it+1} iterations")
            break

    # (5) Total correlation
    h_r = gamma_r + c_r
    g_r = h_r + 1.0
    
    
    
    
    '''
    # -----------------------------
    # Supplied RDF projection (if any)
    # -----------------------------
    g_fixed = np.zeros_like(u_matrix)
    fixed_mask = np.zeros((Ns,Ns), dtype=bool)
    if supplied_data is not None:
        rdf_dict = find_key_recursive(supplied_data, "rdf")
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                pair_key = f"{si}{sj}"
                if pair_key not in rdf_dict:
                    continue
                entry = rdf_dict[pair_key]
                g_sup = np.asarray(entry["g"], float)
                g_fixed[i,j,:,:,:] = g_sup.reshape(Nz,Nz,Nr)
                fixed_mask[i,j] = True

        # Projection loop
        for proj_it in range(n_iter):
            h_old = h_r.copy()

            # (A) Enforce supplied h(r)
            for i in range(Ns):
                for j in range(Ns):
                    if fixed_mask[i,j]:
                        h_r[i,j] = g_fixed[i,j] - 1.0

            # (B) Compute c from OZ (real space)
            c_proj = solve_oz_realspace_planar(h_r, densities, r_grid, z_grid)

            # (C) Freeze supplied c
            for i in range(Ns):
                for j in range(Ns):
                    if fixed_mask[i,j]:
                        c_r[i,j] = c_proj[i,j]

            # (D) Update remaining c from closure
            c_trial = closure_update_c_matrix(
                gamma_r.reshape(Ns,Ns,Nz*Nz*Nr),
                R_ijr.reshape(Nz*Nz*Nr),
                pair_closures,
                u_matrix.reshape(Ns,Ns,Nz*Nz*Nr)
            ).reshape(Ns,Ns,Nz,Nz,Nr)
            for i in range(Ns):
                for j in range(Ns):
                    if not fixed_mask[i,j]:
                        c_r[i,j] = c_trial[i,j]

            # (E) OZ solve with dynamic alpha
            gamma_old_proj = gamma_r.copy()
            gamma_new_proj = solve_oz_matrix_2d(c_r, densities, r_grid, k_grid)
            gamma_r = (1 - alpha) * gamma_old_proj + alpha * gamma_new_proj
            h_r = gamma_r + c_r

            # (F) Convergence (FREE pairs only)
            diff = 0.0
            for i in range(Ns):
                for j in range(Ns):
                    if not fixed_mask[i,j]:
                        diff = max(diff, np.max(np.abs(h_r[i,j] - h_old[i,j])))

            if diff < tol:
                print(f"✅ Projection converged in {proj_it+1} iterations")
                break

        g_r = h_r + 1.0

                
    g_r= h_r+1


    # -----------------------------
    # Export JSON
    # -----------------------------
    if export:
        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_out = {"metadata": {"species": species, "Nz": Nz, "Nr": Nr, "beta": beta}, "pairs": {}}
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                json_out["pairs"][f"{si}{sj}"] = {
                    "R_ijr": R_ijr.tolist(),
                    "g_r": g_r[i,j].tolist(),
                    "h_r": h_r[i,j].tolist(),
                    "c_r": c_r[i,j].tolist(),
                    "gamma_r": gamma_r[i,j].tolist(),
                    "u_r": u_matrix[i,j].tolist(),
                }
        with open(out / f"{filename_prefix}.json","w") as f:
            import json
            json.dump(json_out, f, indent=4)
        print(f"✅ Exported RDF to JSON → {filename_prefix}.json")

    '''
    
    plot_correlation_functions(gamma_r, c_r, r_grid, z_grid, Ns, Nz, Nr, ctx, plot=True)
    
    return {"g_r": g_r, "h_r": h_r, "c_r": c_r, "gamma_r": gamma_r, "u_r": u_matrix}
