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
# -----------------------------
# DST-based Hankel forward/inverse transforms
# -----------------------------

def hankel_forward_dst(f_r, r):
    """
    Forward 3D Hankel transform using DST-I mapping.
    F(k) = (2π Δr / k) * DST₁[r f(r)]
    """
    N = len(r)
    dr = r[1] - r[0]
    Rmax = (N + 1) * dr
    k = np.pi * np.arange(1, N + 1) / Rmax

    x = r * f_r
    X = dst(x, type=1)
    Fk = (2.0 * np.pi * dr / k) * X
    return k, Fk

def hankel_inverse_dst(k, Fk, r):
    """
    Inverse 3D Hankel transform using IDST-I mapping.
    f(r) = (Δk / (4π² r)) * IDST₁[k F(k)]
    """
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
                c_in = -(1+gamma)
                c_out = (1.0 + gamma) * (np.exp(-beta * u) - 1.0)
                c_new[a, b, :] = np.where(core_mask, c_in, c_out)
            elif closure == "HNC":
                c_new[a, b, :] = np.exp(-beta * u + gamma) - gamma - 1.0
            elif closure == "HYBRID":
                
                sigma_ab = sigma_matrix[a, b] if sigma_matrix is not None else 1.0
                core_mask = r < sigma_ab
                c_in = -(1+gamma)
                c_out = np.exp(-beta * u + gamma) - gamma - 1.0
                c_new[a, b, :] = np.where(core_mask, c_in, c_out)
    return c_new

def solve_oz_matrix(c_r_matrix, r, densities):
    N = c_r_matrix.shape[0]
    c_k_matrix, k = hankel_transform_matrix_fast(c_r_matrix, r)
    gamma_k_matrix = np.zeros_like(c_k_matrix)
    rho_matrix = np.diag(densities)
    
    #print (rho_matrix)
    I = np.identity(N)
    eps_reg = 1e-12

    for ik in range(len(k)):
        Ck = c_k_matrix[:, :, ik]
        num = Ck @ rho_matrix @ Ck
        A = I - Ck @ rho_matrix + eps_reg * I
        gamma_k_matrix[:, :, ik] = np.linalg.solve(A, num)

    gamma_r_matrix = inverse_hankel_transform_matrix_fast(gamma_k_matrix, k, r)
    return gamma_r_matrix

import numpy as np

def multi_component_oz_solver_alpha(
    r, pair_closures, densities, beta, sigma_matrix=None, u_soft_matrix=None,
    n_iter=10000, tol=1e-12, alpha=0.1, alpha_min=1e-6, alpha_max=0.5, recovery_patience=5
):
    """
    Multi-component Ornstein–Zernike (OZ) solver with adaptive alpha-mixing and recovery mechanism.
    """

    N = pair_closures.shape[0]
    Nr = len(r)

    if u_soft_matrix is None:
        raise ValueError("u_soft_matrix must be provided to multi_component_oz_solver_alpha().")

    gamma_r = np.zeros_like(u_soft_matrix)
    c_r = np.zeros_like(u_soft_matrix)

    # store last stable values
    last_gamma = gamma_r.copy()
    last_c = c_r.copy()
    last_diff = np.inf
    failure_count = 0

    print(f"\nStarting OZ solver with adaptive α (initial = {alpha}), tol = {tol}, max_iter = {n_iter}")
    print("Iter | Δγ(max) | α")

    for step in range(n_iter):
        try:
            # --- closure update ---
            c_r = closure_update_c_matrix(gamma_r, r, beta, pair_closures, u_soft_matrix, sigma_matrix)

            # --- OZ equation solve ---
            gamma_new = solve_oz_matrix(c_r, r, densities)

            # --- check for None or invalid results ---
            if gamma_new is None or not np.all(np.isfinite(gamma_new)):
                raise ValueError("gamma_new returned None or contains NaN/Inf")

            # --- convergence measure ---
            delta_gamma = gamma_new - gamma_r
            diff = np.max(np.abs(delta_gamma))

            # --- alpha-mixing update ---
            gamma_r = (1 - alpha) * gamma_r + alpha * gamma_new

            # --- print progress ---
            if step % 10 == 0 or diff < tol:
                print(f"{step:5d} | {diff:9.3e} | {alpha:6.3f}")

            # --- check convergence ---
            if diff < tol:
                print(f"\n✅ Converged in {step+1} iterations (Δγ = {diff:.3e}, α = {alpha:.3f})")
                break

            # --- dynamic alpha tuning ---
            if diff < last_diff:
                # iteration improving, slightly increase alpha (within safe range)
                alpha = min(alpha * 1.05, alpha_max)
                last_gamma = gamma_r.copy()
                last_c = c_r.copy()
                last_diff = diff
                failure_count = 0
            else:
                # worsening or unstable → decrease alpha and revert
                alpha = max(alpha * 0.1, alpha_min)
                gamma_r = last_gamma.copy()
                c_r = last_c.copy()
                failure_count += 1
                print(f"⚠️  Step {step}: Instability detected, reducing α to {alpha:.3e}")

                if failure_count >= recovery_patience:
                    print("\n❌ Repeated instability detected. Exiting early with last stable solution.")
                    break

        except Exception as e:
            # handle unexpected numerical failures
            alpha = max(alpha * 0.1, alpha_min)
            gamma_r = last_gamma.copy()
            c_r = last_c.copy()
            failure_count += 1
            print(f"⚠️  Step {step}: Exception '{e}', reducing α to {alpha:.3e}")

            if failure_count >= recovery_patience:
                print("\n❌ Repeated failures. Returning last stable result.")
                break

    else:
        print(f"\n⚠️ Warning: OZ solver did not converge after {n_iter} iterations (Δγ = {diff:.3e}).")

    # --- Final structural correlations ---
    h_r = gamma_r + c_r
    g_r = h_r + 1.0

    return c_r, gamma_r, g_r



# -----------------------------
# vk integral
# -----------------------------

def compute_vk_from_g_phi(r, g_r_matrix, phi_r_matrix):
    N = g_r_matrix.shape[0]
    vk = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            integrand = g_r_matrix[a, b, :] * phi_r_matrix[a, b, :] * 4.0 * np.pi * r**2
            vk[a, b] = np.trapz(integrand, r)
    return vk

# -----------------------------
# RDF master (accepts precomputed u_matrix and sigma_matrix)
# -----------------------------

def rdf(species,
        pair_closures,
        densities,
        r,
        beta=1.0,
        n_iter=10000,
        tol=1e-6,
        alpha=0.01,
        pre_u_matrix=None,
        pre_sigma_matrix=None):
    """
    Compute c_r, gamma_r, g_r and vk using a precomputed u_matrix (N,N,Nr) and sigma_matrix (N,N).
    pair_closures: numpy array (N,N) of closure strings ('HNC','PY','HYBRID')
    densities: array-like length N
    r: radial grid
    """
    N = len(species)
    if pre_u_matrix is None or pre_sigma_matrix is None:
        raise ValueError("Please provide pre_u_matrix and pre_sigma_matrix to rdf().")

    c_r, gamma_r, g_r = multi_component_oz_solver_alpha(
        r,
        pair_closures,
        densities,
        beta,
        sigma_matrix=pre_sigma_matrix,
        u_soft_matrix=pre_u_matrix,
        n_iter=n_iter,
        tol=tol,
        alpha=alpha,
    )

    vk = compute_vk_from_g_phi(r, g_r, pre_u_matrix)
    return vk, c_r, gamma_r, g_r

# -----------------------------
# Entry function vk_rdf: build u_matrix using ppi and call rdf()
# -----------------------------

def vk_rdf(ctx, beta=1.0, densities=None, r_min=5.0/400, r_max=5.0, Nr=400):
    """
    Top-level: build u_matrix from meanfield splitter using pair_potential_isotropic (ppi),
    build sigma_matrix from hard-core splitter, choose closures from hc flags,
    and call rdf() with precomputed matrices.
    """
    if densities is None:
        raise ValueError("Please provide densities (array-like) to vk_rdf().")

    # Load splitters
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)

    species = pot_data["species"]
    interactions_by_level = pot_data["interactions"]
    if pot_data["interactions"] is None:
        interactions_by_level = raw_potentials
        
    
    
    levels = ["primary", "secondary", "tertiary"]
    N = len(species)

    # radial grid
    r = np.linspace(r_min, r_max, Nr)

   
    # -----------------------------
    # sigma matrix from hc_data (Lorentz additivity)
    # -----------------------------
    # -----------------------------
    # sigma matrix from hc_data (Lorentz additivity + flag filtering)
    # -----------------------------

    if hc_data is None:
        # no hard-core data: everything zero
        sigma_eff = np.zeros(len(species))
        flags = np.zeros(len(species), dtype=int)
    else:
        sigma_eff = []
        flags = []
        for sp in species:
            if sp in hc_data:
                sigma_eff.append(hc_data[sp].get("sigma_eff", 0.0))
                flags.append(hc_data[sp].get("flag", 0))
            else:
                sigma_eff.append(0.0)
                flags.append(0)
        sigma_eff = np.array(sigma_eff)
        flags = np.array(flags, dtype=int)

    # Lorentz additivity rule (pairwise arithmetic mean)
    sigma_matrix = 0.5 * (sigma_eff[:, None] + sigma_eff[None, :])

    # Zero out sigma_matrix entries where BOTH species have flag == 0
    flag_mask = (flags[:, None] == 0) & (flags[None, :] == 0)
    sigma_matrix[flag_mask] = 0.0
    
    print ("\n\n\n\n\n")
    print (sigma_matrix)



    # initialize u_matrix
    u_matrix = np.zeros((N, N, Nr))

    # build potentials directly from ppi for each level; superimpose contributions
    for level in levels:
        level_dict = interactions_by_level.get(level, {}) or {}
        if not level_dict:
            continue
        for pair_key, params in level_dict.items():
            if len(pair_key) != 2:
                continue
            a, b = pair_key[0], pair_key[1]
            try:
                i = species.index(a)
                j = species.index(b)
            except ValueError:
                # species not found — skip
                continue

            # prepare potential dict exactly as provided by splitter (no manual extraction)
            potential_dict = params.copy() if isinstance(params, dict) else dict(params)
            # ensure cutoff exists so we can evaluate on appropriate r-space
            cutoff = potential_dict.get("cutoff", r_max)
            # get vectorized potential function from the centralized generator
            V_func = ppi(potential_dict)

            # evaluate on a local r_space (dense enough)
            r_space = np.linspace(0.0, r_max, Nr)
            V_r_local = V_func(r_space)

            # interpolate (if necessary) onto master r
            if not np.allclose(r_space, r):
                interp = interp1d(r_space, V_r_local, bounds_error=False, fill_value="extrapolate")
                V_r = interp(r)
            else:
                V_r = V_r_local

            # superimpose and enforce symmetry
            u_matrix[i, j, :] += V_r
            if i != j:
                u_matrix[j, i, :] += V_r

    # determine pair_closures from hc flags (HYBRID if flagged, else HNC as default)
    
    pair_closures = np.empty((N, N), dtype=object)
    # choose closure policy: if any hc flag==1, use 'PY' for all; else 'HNC'
    
    for i in range(N):
        for j in range(N):
            if (sigma_matrix[i][j] == 0.0):
                pair_closures[i][j] = 'HNC'
            else:
                pair_closures[i][j] = 'PY'
                
                
    print (pair_closures)
    
    # call rdf with precomputed u_matrix
    vk, c_r, gamma_r, g_r = rdf( species=species, pair_closures=pair_closures, densities=densities, r=r, beta=beta, pre_u_matrix=u_matrix, pre_sigma_matrix=sigma_matrix,)

        # -----------------------------
    # Export RDF (g(r)) plots for all species pairs
    # -----------------------------
        # -----------------------------
    # Export RDF (g(r)) plots for all species pairs
    # -----------------------------
    import matplotlib.pyplot as plt
    import re

    # ensure plot directory exists
    plots_dir = Path(ctx.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create a density-based suffix for filename
    # e.g. [0.53, 0.004, 0.098] → "dens_0.53_0.004_0.098"
    densities_str = "_".join(f"{d:.4f}" for d in np.atleast_1d(densities))
    # sanitize any invalid filename characters
    densities_str = re.sub(r"[^0-9A-Za-z_.-]", "_", densities_str)

    N = len(species)
    fig, axes = plt.subplots(N, N, figsize=(4*N, 3.5*N), dpi=120, sharex=True, sharey=True)

    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            ax = axes[i, j] if N > 1 else axes
            # plot g(r)
            ax.plot(r, g_r[i, j, :], color='tab:blue', lw=1.8, label='g(r)')
            # plot u(r) on secondary axis
            ax2 = ax.twinx()
            ax2.plot(r, u_matrix[i, j, :], color='tab:red', lw=1.2, ls='--', label='u(r)')
            ax.set_title(f"{sp_i}-{sp_j}", fontsize=10)
            if i == N - 1:
                ax.set_xlabel("r")
            if j == 0:
                ax.set_ylabel("g(r)")
            ax.grid(alpha=0.3)
            ax.set_xlim(r[0], r[-1])
            ax.set_ylim(0, np.max(g_r)*1.1)
            ax2.set_ylabel("u(r)", color='tab:red', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.suptitle("Pair Correlation g(r) and Potential u(r) for All Species Pairs", fontsize=14, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # save combined plot with densities in the filename
    out_path = plots_dir / f"rdf_all_pairs_matrix_{densities_str}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved n×n RDF + potential plot grid to {out_path}")




    return {"species": species, "vk": vk}

