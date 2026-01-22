import json
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials


def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None


def second_virial(
    ctx,
    virial_config,
    on="splitted",
    export=True,
    filename_prefix="second_virial_coefficient",
    r_max_factor=6.0,
    nr=8192,
    n_lambda=128,
    beta_scale=1.0,
):
    """
    Research-grade computation of:
      - Second virial coefficients B2^{ij}
      - Integrated strength via thermodynamic (lambda) integration
    using the exact low-density limit with hard-core + mean-field splitting.
    """

    # -----------------------------
    # Configuration
    # -----------------------------
    virial_block = find_key_recursive(virial_config, "virial")
    if virial_block is None:
        raise KeyError("Missing 'virial' block")

    species = find_key_recursive(virial_config, "species")
    beta = virial_block.get("beta", beta_scale)
    n = len(species)

    # -----------------------------
    # Generate potentials
    # -----------------------------
    hc_data = hard_core_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)
    mf_data = meanfield_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)
    total_data = total_potentials(ctx=ctx, hc_source=hc_data, mf_source=mf_data, export_files=True)
    raw_data = raw_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)

    # -----------------------------
    # Radial grid
    # -----------------------------
    sigma = np.asarray(hc_data["sigma"])
    sigma_max = np.max(sigma)
    r_max = r_max_factor * sigma_max
    r = np.linspace(1e-12, r_max, nr)
    dr = r[1] - r[0]

    # -----------------------------
    # Select potential
    # -----------------------------
    if on == "splitted":
        potential_dict = mf_data["potentials"]
        u_attr = np.zeros((n, n, nr))
        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                interp_u = interp1d(
                    pdata["r"], pdata["U"], bounds_error=False, fill_value=0.0, assume_sorted=True
                )
                u_attr_ij = beta * interp_u(r)
                u_attr[i, j, :] = u_attr_ij
                u_attr[j, i, :] = u_attr_ij

        # -----------------------------
        # Lambda grid
        # -----------------------------
        lam = np.linspace(0.0, 1.0, n_lambda)
        dlam = lam[1] - lam[0]

        # -----------------------------
        # Allocate results
        # -----------------------------
        B2 = np.zeros((n, n))
        integrated_strength = np.zeros((n, n))

        # -----------------------------
        # Core computation
        # -----------------------------
        for i in range(n):
            for j in range(i, n):

                # ---- Extract potentials ----
                u_attr_ij = u_attr[i, j]
                sigma_ij = sigma[i, j]
                u_hc = np.zeros_like(r)
                u_hc[r < sigma_ij] = 1e12  # large repulsive core

                # ---- Second virial coefficient: f = exp(-beta u_total) - 1 ----
                u_total = u_hc + u_attr_ij
                f = np.exp(-u_total) - 1.0
                integrand = 4.0 * np.pi * r**2 * f
                B2_ij = -0.5 * np.trapz(integrand, r)

                # ---- Integrated strength (thermodynamic integration) ----
                # lambda perturbs only the attractive part
                # Exponential sees: u_hc + lambda * u_attr
                lambda_integrated = np.zeros_like(r)
                for lam_k in lam:
                    gl = np.exp(-(u_hc + lam_k * u_attr_ij))
                    lambda_integrated += gl * u_attr_ij * dlam  # g_lambda * u_attr

                # Radial integration
                strength_integrand = 4.0 * np.pi * r**2 * lambda_integrated
                I_ij = np.trapz(strength_integrand, r)

                # Symmetric assignment
                B2[i, j] = B2[j, i] = B2_ij
                integrated_strength[i, j] = integrated_strength[j, i] = I_ij

        # -----------------------------
        # Export results
        # -----------------------------
        if export:
            out = Path(ctx.scratch_dir)
            out.mkdir(parents=True, exist_ok=True)

            data = {
                "metadata": {
                    "species": species,
                    "beta": beta,
                    "r_max": r_max,
                    "nr": nr,
                    "n_lambda": n_lambda,
                },
                "pairs": {},
            }

            for i, si in enumerate(species):
                for j, sj in enumerate(species):
                    key = f"{si}{sj}"
                    data["pairs"][key] = {
                        "B2": float(B2[i, j]),
                        "integrated_strength": float(integrated_strength[i, j]),
                    }

            path = out / f"{filename_prefix}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"✅ Second virial + integrated strength exported → {path}")

        return B2, integrated_strength

            
        
        
        
        
        
        
        
        
        
    else:
        # -----------------------------
        # Build beta * u_ij(r) from raw potentials
        # -----------------------------
        potential_dict = raw_data["potentials"]
        u = np.zeros((n, n, nr))

        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]

                # Determine the key for the pair
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                # Interpolate potential on radial grid
                interp_u = interp1d(
                    pdata["r"],
                    pdata["U"],
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )

                u_ij = beta * interp_u(r)
                u[i, j, :] = u_ij
                u[j, i, :] = u_ij

        # -----------------------------
        # Lambda grid
        # -----------------------------
        lam = np.linspace(0.1, 1.0, n_lambda)
        dlam = lam[1] - lam[0]

        # -----------------------------
        # Allocate results
        # -----------------------------
        B2 = np.zeros((n, n))
        integrated_strength = np.zeros((n, n))

        # -----------------------------
        # Core computation
        # -----------------------------
        for i in range(n):
            for j in range(i, n):

                uij = u[i, j]

                # ---- Second virial coefficient ----
                f = np.exp(-uij) - 1.0
                integrand = 4.0 * np.pi * r**2 * f
                B2_ij = -0.5 * np.trapz(integrand, r)

                # ---- Integrated strength (thermodynamic integration) ----
                # g_lambda(r) = exp(-lambda * beta u)
                gl = np.exp(-lam[:, None] * uij[None, :])
                lambda_integrand = gl * uij[None, :]
                lambda_integrated = np.trapz(lambda_integrand, lam, axis=0)

                strength_integrand = 4.0 * np.pi * r**2 * lambda_integrated
                I_ij = np.trapz(strength_integrand, r)

                # Symmetric assignment
                B2[i, j] = B2[j, i] = B2_ij
                integrated_strength[i, j] = integrated_strength[j, i] = I_ij

        # -----------------------------
        # Export results
        # -----------------------------
        if export:
            out = Path(ctx.scratch_dir)
            out.mkdir(parents=True, exist_ok=True)

            data = {
                "metadata": {
                    "species": species,
                    "beta": beta,
                    "r_max": r_max,
                    "nr": nr,
                    "n_lambda": n_lambda,
                },
                "pairs": {},
            }

            for i, si in enumerate(species):
                for j, sj in enumerate(species):
                    key = f"{si}{sj}"
                    data["pairs"][key] = {
                        "B2": float(B2[i, j]),
                        "integrated_strength": float(integrated_strength[i, j]),
                    }

            path = out / f"{filename_prefix}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"✅ Second virial + integrated strength exported → {path}")

        return B2, integrated_strength

