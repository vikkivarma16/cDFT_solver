def mf_weights_1d(ctx):
    """
    Compute mean-field weights in k-space using the mean-field potential splitter.
    - Extract mean-field potentials via meanfield_potentials(ctx, mode="meanfield")
    - Build u_mf_matrix (N,N,Nr) by summing mean-field layers for each pair
    - For each pair (i,j) compute Vk(k) = ∫ V(r) cos(2π k r) dr using scipy.integrate.quad
    - Save weight files and plots to ctx.scratch_dir / ctx.plots_dir
    """
    import numpy as np
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    from scipy import integrate
    from scipy.interpolate import interp1d

    # project-specific imports (paths used in your repo)
    from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
    from cdft_solver.generators.potential_splitter.generator_potential_total import raw_potentials
    from cdft_solver.generators.potential.generator_pair_potential_one_d import (
        pair_potential_one_d,
        pair_potential_one_d_integrant,
    )

    # -------------------------
    # Prepare directories & files
    # -------------------------
    scratch = Path(ctx.scratch_dir)
    plots_dir = Path(ctx.plots_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # files you already use for r and k grids
    r_space_file = scratch / "supplied_data_r_space.txt"
    k_space_file = scratch / "supplied_data_k_space.txt"

    # load r and k arrays (expect two-column text files where first col is grid)
    r_space = np.loadtxt(r_space_file)
    k_space = np.loadtxt(k_space_file)
    r = r_space[:, 0]
    k_values = k_space[:, 0]
    Nr = len(r)

    # -------------------------
    # Extract mean-field potentials
    # -------------------------
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    if not pot_data or pot_data.get("interactions") is None:
        # fallback
        interactions_by_level = raw_potentials
    else:
        interactions_by_level = pot_data.get("interactions", {}) or raw_potentials

    species = pot_data.get("species", [])
    if not species:
        raise RuntimeError("meanfield_potentials returned no species list.")

    levels = ["primary", "secondary", "tertiary"]
    N = len(species)

    # -------------------------
    # Build MF potential matrix u_mf_matrix (N,N,Nr)
    # -------------------------
    u_mf_matrix = np.zeros((N, N, Nr), dtype=float)

    for level in levels:
        level_dict = interactions_by_level.get(level, {}) or {}
        if not level_dict:
            continue
        for pair_key, params in level_dict.items():
            # pair_key expected "AB" two-letter string
            if not isinstance(pair_key, str) or len(pair_key) != 2:
                continue
            a, b = pair_key[0], pair_key[1]
            try:
                i = species.index(a)
                j = species.index(b)
            except ValueError:
                # species not present in this meanfield dataset
                continue

            # ensure we have a dict copy
            potdict = params.copy() if isinstance(params, dict) else dict(params)

            # ensure common potential fields exist for pair_potential_one_d
            if "cutoff" not in potdict:
                potdict["cutoff"] = float(np.max([r[-1], 5.0]))

            # Generate vectorized potential V(r) from pair_potential_one_d
            V_vec = pair_potential_one_d(potdict)

            # Evaluate on a local r-space (use same Nr but starting at 0 to match generator expectations)
            local_r = np.linspace(0.0, r[-1], Nr)
            V_local = V_vec(local_r)

            # Interpolate onto master r (master r may start >0; we still extrapolate if needed)
            if not np.allclose(local_r, r):
                interp = interp1d(local_r, V_local, bounds_error=False, fill_value="extrapolate")
                V_on_master = interp(r)
            else:
                V_on_master = V_local

            # Add contribution to u_mf_matrix (symmetrize)
            u_mf_matrix[i, j, :] += V_on_master
            if i != j:
                u_mf_matrix[j, i, :] += V_on_master

    # -------------------------
    # Hankel transform (direct integral) per pair using quad
    # V_k(k) = ∫ V(r) cos(2π k r) dr
    # Use pair_potential_one_d_integrant when possible for scalar integrand
    # -------------------------
    weight_matrix_k = [[None] * N for _ in range(N)]

    # integration settings
    quad_opts = dict(limit=20000, epsabs=1e-12, epsrel=1e-10)

    # We'll pick integration bounds per pair: use +/- max_cutoff or +/-50*sigma_est
    # If potentials include 'sigma' we use it, else fallback to r[-1] or 50.
    for i in range(N):
        for j in range(i, N):
            # build composite V(r) for this pair as a python callable for scalar r
            V_array = u_mf_matrix[i, j, :]

            # attempt to construct scalar integrand from pair_potential_one_d_integrant for each layer:
            # fallback: use interpolation of V_array
            use_scalar_integrant = False
            # choose sigma estimate from data if available by inspecting any pot in interactions_by_level
            sigma_est = 1.0
            # attempt to find sigma in any level params for this pair
            for level in levels:
                lvl = interactions_by_level.get(level, {}) or {}
                p = lvl.get(species[i] + species[j]) or lvl.get(species[j] + species[i])
                if isinstance(p, dict) and "sigma" in p:
                    sigma_est = float(p.get("sigma", sigma_est))
                    break

            # integration window
            Rcut = max(r[-1], 50.0 * sigma_est)

            # create interpolator for V(r) (to evaluate at scalar r for quad)
            V_interp = interp1d(r, V_array, bounds_error=False, fill_value=0.0)

            # integrand uses absolute r (potentials defined for r>=0)
            def integrand_scalar(scalar_r, k_val):
                # use interpolated composite V(r) at abs(scalar_r)
                return float(V_interp(abs(scalar_r))) * np.cos(2.0 * np.pi * k_val * scalar_r)

            Vk_list = []
            for k_val in k_values:
                # integrate from -Rcut to +Rcut (potential even => integrand even, but we stick with full integral)
                val, err = integrate.quad(lambda rr: integrand_scalar(rr, k_val),
                                         -Rcut, Rcut, **quad_opts)
                Vk_list.append(complex(val))
            Vk = np.array(Vk_list, dtype=complex)

            # save
            weight_matrix_k[i][j] = Vk
            if i != j:
                weight_matrix_k[j][i] = Vk  # symmetry

            # write file and plot (immediately)
            pair_tag = f"{species[i]}{species[j]}"
            out_file = scratch / f"supplied_data_weight_MF_k_space_{pair_tag}.txt"
            np.savetxt(out_file, Vk)

            # plotting
            plt.figure(figsize=(7, 5))
            plt.plot(k_values, Vk.real, label="Re[V(k)]")
            plt.plot(k_values, Vk.imag, label="Im[V(k)]")
            plt.plot(k_values, np.abs(Vk), "--", label="|V(k)|")
            plt.xlabel(r"$k$")
            plt.ylabel(r"$V(k)$")
            plt.title(f"Mean-field weight in k-space ({pair_tag})")
            plt.legend()
            plt.grid(True, ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plots_dir / f"vis_weight_MF_k_space_{pair_tag}.png", dpi=300)
            plt.close()

    print("\n\n✅ Mean field weight calculated from meanfield_potentials, exported weights + plots.\n")
    return {"species": species, "weights_k": weight_matrix_k, "u_mf_r": u_mf_matrix}

