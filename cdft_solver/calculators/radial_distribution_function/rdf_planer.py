import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.special import j0

from .closer import closure_update_c_matrix
from .rdf_isotropic import find_key_recursive

# -------------------------------------------------
# Hankel DST transforms (radial)
# -------------------------------------------------
def hankel_transform_2d(f_r, r, k_grid):
    """Apply Hankel transform for cylindrical coordinates along r."""
    dr = r[1] - r[0]
    return 2 * np.pi * (j0(np.outer(k_grid, r)) @ (r * f_r)) * dr

def inverse_hankel_transform_2d(F_k, r, k_grid):
    dk = k_grid[1] - k_grid[0]
    return (j0(np.outer(k_grid, r)).T @ (k_grid * F_k)) * dk / (2*np.pi)

# -------------------------------------------------
# Solve OZ in k-space
# -------------------------------------------------
def solve_oz_matrix_2d(c_r_matrix, densities, r, k_grid):
    """2D OZ solver along r for each plane z_i, z_j."""
    Ns, _, Nz, _, Nr = c_r_matrix.shape
    Ck = np.zeros_like(c_r_matrix)
    gamma_k = np.zeros_like(c_r_matrix)

    # Forward Hankel along r
    for a in range(Ns):
        for b in range(Ns):
            for i in range(Nz):
                for j in range(Nz):
                    Ck[a,b,i,j,:] = hankel_transform_2d(c_r_matrix[a,b,i,j,:], r, k_grid)

    # OZ in k-space (plane-wise)
    rho_diag = np.diag(np.repeat(densities, Nz))
    Nd = Ns * Nz
    I = np.eye(Nd)

    for ik in range(len(k_grid)):
        C_big = Ck[..., ik].transpose(0,2,1,3).reshape(Nd, Nd)
        M = I - C_big @ rho_diag
        gamma_k_flat = np.linalg.solve(M, (C_big @ rho_diag @ C_big))
        gamma_k[..., ik] = gamma_k_flat.reshape(Ns, Nz, Ns, Nz).transpose(0,2,1,3)

    # Inverse Hankel
    gamma_r = np.zeros_like(c_r_matrix)
    for a in range(Ns):
        for b in range(Ns):
            for i in range(Nz):
                for j in range(Nz):
                    gamma_r[a,b,i,j,:] = inverse_hankel_transform_2d(gamma_k[a,b,i,j,:], r, k_grid)

    return gamma_r

# -------------------------------------------------
# Main 2D RDF solver
# -------------------------------------------------
def rdf_planar(
    ctx,
    rdf_config,
    r_space,           # shape (N,3), N = Nz*Nr
    potential_dict,
    densities,
    sigma=None,
    supplied_data=None,
    export=False,
    plot=True,
    filename_prefix="rdf_2d"
):
    """
    Planar/cylindrical 2D RDF solver
    """

    # -----------------------------
    # Extract RDF parameters
    # -----------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    if rdf_block is None:
        raise KeyError("No 'rdf' key found in rdf_config")

    params = rdf_config["rdf_parameters"]
    species = params["species"]
    Ns = len(species)

    beta = rdf_block.get("beta", 1.0)
    tol = rdf_block.get("tolerance", 1e-6)
    n_iter = rdf_block.get("max_iteration", 10000)
    alpha_max = rdf_block.get("alpha_max", 0.05)

    # -----------------------------
    # Extract cylindrical grids
    # -----------------------------
    NzNr = r_space.shape[0]
    Nz = len(np.unique(r_space[:,0]))   # assuming z first column
    Nr = len(np.unique(r_space[:,1]))   # assuming r second column
    z_grid = np.unique(r_space[:,0])
    r_grid = np.unique(r_space[:,1])

    # Radial Hankel k-grid
    k_grid = np.linspace(0.0, 20.0, Nr)

    # -----------------------------
    # Closure matrix
    # -----------------------------
    closure_cfg = rdf_block.get("closure", {})
    pair_closures = np.empty((Ns,Ns), dtype=object)
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            key = f"{si}{sj}"
            pair_closures[i,j] = closure_cfg[key]

    # -----------------------------
    # Potential matrix
    # -----------------------------
    # z_grid: shape (Nz,)
    # r_grid: shape (Nr,)

    Zij = z_grid[:, None] - z_grid[None, :]     # (Nz, Nz)
    R_ijr = np.sqrt(Zij[:, :, None]**2 + r_grid[None, None, :]**2)
    # shape: (Nz, Nz, Nr)

    u_matrix = np.zeros((Ns, Ns, Nz, Nz, Nr))
    for a, sa in enumerate(species):
        for b, sb in enumerate(species):

            pdata = potential_dict.get((sa, sb), potential_dict.get((sb, sa)))
            if pdata is None:
                raise KeyError(f"Missing potential for pair {sa}-{sb}")

            ru = np.asarray(pdata["r"], float)
            uu = np.asarray(pdata["u"], float)

            interp_u = interp1d(
                ru,
                uu,
                bounds_error=False,
                fill_value=0.0
            )

            # evaluate on full (Nz, Nz, Nr) distance grid
            u_matrix[a, b] = beta * interp_u(R_ijr)


    # -----------------------------
    # Sigma matrix
    # -----------------------------
    sigma_matrix = np.zeros((Ns,Ns)) if sigma is None else sigma

    # -----------------------------
    # Initialize correlation functions
    # -----------------------------
    gamma_r = np.zeros_like(u_matrix)
    c_r = np.zeros_like(u_matrix)

    # -----------------------------
    # Supplied data processing
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

    # -----------------------------
    # Main iterative OZ solver loop
    # -----------------------------
    for it in range(n_iter):
        gamma_old = gamma_r.copy()

        # Closure update (only non-frozen)
        c_trial = closure_update_c_matrix(gamma_r.reshape(Ns,Ns,Nz*Nz*Nr), R_ijr.reshape(Nz*Nz*Nr), pair_closures, u_matrix.reshape(Ns,Ns,Nz*Nz*Nr))
        c_trial = c_trial.reshape(Ns,Ns,Nz,Nz,Nr)
        for i in range(Ns):
            for j in range(Ns):
                if not fixed_mask[i,j]:
                    c_r[i,j,:,:,:] = c_trial[i,j,:,:,:]

        # OZ solve
        gamma_r = solve_oz_matrix_2d(c_r, densities, r_grid, k_grid)

        # Convergence check
        err = np.max(np.abs(gamma_r - gamma_old))
        if it % 10 == 0:
            print(f"Iteration {it} | Δγ = {err:.3e}")
        if err < tol:
            print(f"✅ Converged after {it+1} iterations")
            break

    # Total correlation
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

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
                    "r_space": r_space.tolist(),
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

    return {"g_r": g_r, "h_r": h_r, "c_r": c_r, "gamma_r": gamma_r, "u_r": u_matrix}

