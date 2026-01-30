import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
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


def compute_B2_with_epsilon(epsilon, r, u_hc, u_attr):
    u_tot = u_hc + epsilon * u_attr
    f = np.exp(-u_tot) - 1.0
    integrand = 4.0 * np.pi * r**2 * f
    return -0.5 * np.trapz(integrand, r)


def second_virial_epsilon_calibration(
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
      - Calibrated epsilon such that B2(epsilon) = B2_target
    """

    # -----------------------------
    # Configuration
    # -----------------------------
    virial_block = find_key_recursive(virial_config, "virial")
    if virial_block is None:
        raise KeyError("Missing 'virial' block")

    species = find_key_recursive(virial_config, "species")
    beta = virial_block.get("beta", beta_scale)
    B2_target = virial_block.get("B2_target", {})
    n = len(species)

    # -----------------------------
    # Generate potentials
    # -----------------------------
    hc_data = hard_core_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)
    mf_data = meanfield_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)
    raw_data = raw_potentials(ctx=ctx, input_data=virial_config, grid_points=nr, export_files=True)

    # -----------------------------
    # Radial grid
    # -----------------------------
    sigma = np.asarray(hc_data["sigma"])
    r_max = r_max_factor * np.max(sigma)
    r = np.linspace(1e-12, r_max, nr)

    # ======================================================================
    # SPLITTED MODE (ε CALIBRATION ENABLED)
    # ======================================================================
    if on == "splitted":

        potential_dict = mf_data["potentials"]
        u_attr = np.zeros((n, n, nr))

        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                interp_u = interp1d(
                    pdata["r"], pdata["U"],
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )

                u_attr_ij = beta * interp_u(r)
                u_attr[i, j] = u_attr[j, i] = u_attr_ij

        lam = np.linspace(0.0, 1.0, n_lambda)
        dlam = lam[1] - lam[0]

        B2 = np.zeros((n, n))
        integrated_strength = np.zeros((n, n))
        epsilon_calibrated = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):

                u_attr_ij = u_attr[i, j]
                sigma_ij = sigma[i, j]

                u_hc = np.zeros_like(r)
                u_hc[r < sigma_ij] = 1e12

                # ---- B2 at epsilon = 1 ----
                B2_ij = compute_B2_with_epsilon(1.0, r, u_hc, u_attr_ij)

                # ---- Integrated strength ----
                lambda_integrated = np.zeros_like(r)
                for lam_k in lam:
                    gl = np.exp(-(u_hc + lam_k * u_attr_ij))
                    lambda_integrated += gl * u_attr_ij * dlam

                I_ij = np.trapz(4.0 * np.pi * r**2 * lambda_integrated, r)

                # ---- Target B2 ----
                key = f"{species[i]}{species[j]}"
                key_rev = f"{species[j]}{species[i]}"
                B2_tgt = B2_target.get(key, B2_target.get(key_rev, None))

                if B2_tgt is None:
                    eps_ij = 1.0
                else:
                    def root_fn(eps):
                        return compute_B2_with_epsilon(eps, r, u_hc, u_attr_ij) - B2_tgt

                    eps_ij = brentq(root_fn, 0.0, 10.0)

                B2[i, j] = B2[j, i] = B2_ij
                integrated_strength[i, j] = integrated_strength[j, i] = I_ij
                epsilon_calibrated[i, j] = epsilon_calibrated[j, i] = eps_ij

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
                        "B2_target": float(B2_target.get(key, np.nan)),
                        "integrated_strength": float(integrated_strength[i, j]),
                        "epsilon_calibrated": float(epsilon_calibrated[i, j]),
                    }

            path = out / f"{filename_prefix}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"✅ B2 + ε calibration exported → {path}")

        return B2, integrated_strength, epsilon_calibrated

    # ======================================================================
    # RAW MODE (NO ε CALIBRATION)
    # ======================================================================
    else:
        potential_dict = raw_data["potentials"]
        u = np.zeros((n, n, nr))

        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                interp_u = interp1d(
                    pdata["r"], pdata["U"],
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )

                u_ij = beta * interp_u(r)
                u[i, j] = u[j, i] = u_ij

        lam = np.linspace(0.1, 1.0, n_lambda)

        B2 = np.zeros((n, n))
        integrated_strength = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):

                uij = u[i, j]
                f = np.exp(-uij) - 1.0
                B2_ij = -0.5 * np.trapz(4.0 * np.pi * r**2 * f, r)

                gl = np.exp(-lam[:, None] * uij[None, :])
                lambda_integrated = np.trapz(gl * uij[None, :], lam, axis=0)
                I_ij = np.trapz(4.0 * np.pi * r**2 * lambda_integrated, r)

                B2[i, j] = B2[j, i] = B2_ij
                integrated_strength[i, j] = integrated_strength[j, i] = I_ij

        return B2, integrated_strength

