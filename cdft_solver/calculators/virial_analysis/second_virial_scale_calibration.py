import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from pathlib import Path

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials


# ============================================================
# Utilities
# ============================================================

def find_key_recursive(d, key):
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None


def compute_B2_scaled(scale, r, u_total):
    """
    Second virial coefficient using a globally scaled total potential:
        u_scaled(r) = scale * u_total(r)
    """
    f = np.exp(-scale * u_total) - 1.0
    integrand = 4.0 * np.pi * r**2 * f
    return -0.5 * np.trapz(integrand, r)


def unpack_scale_vector(scale_vec, pair_list, n):
    scale_mat = np.ones((n, n))
    k = 0
    for (i, j) in pair_list:
        scale_mat[i, j] = scale_mat[j, i] = scale_vec[k]
        k += 1
    return scale_mat


# ============================================================
# MAIN CALIBRATION ROUTINE
# ============================================================

def second_virial_scale_calibration(
    ctx,
    virial_config,
    B2_target,
    on="splitted",
    export=True,
    filename_prefix="second_virial_scale_calibration",
    r_max_factor=6.0,
    nr=8192,
    beta_scale=1.0,
):
    """
    Calibrate pairwise scalar factors f_ij such that

        B2_ij[ f_ij * u_total^{ij}(r) ] = B2_target^{ij}

    The scaling is applied to the FULL supplied potential.
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

    B2_target = np.asarray(B2_target, float)

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
    r_max = r_max_factor * np.max(sigma)
    r = np.linspace(1e-12, r_max, nr)

    # ============================================================
    # SPLITTED MODE
    # ============================================================
    if on == "splitted":

        potential_dict = mf_data["potentials"]
        u_attr = np.zeros((n, n, nr))

        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                interp_u = interp1d(
                    pdata["r"],
                    pdata["U"],
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )

                u_attr_ij = beta * interp_u(r)
                u_attr[i, j] = u_attr[j, i] = u_attr_ij

        # ---- build total potentials ----
        u_total = np.zeros((n, n, nr))
        for i in range(n):
            for j in range(i, n):
                u_hc = np.zeros_like(r)
                u_hc[r < sigma[i, j]] = 1e12
                u_tot_ij = u_hc + u_attr[i, j]
                u_total[i, j] = u_total[j, i] = u_tot_ij

    # ============================================================
    # RAW MODE
    # ============================================================
    else:
        potential_dict = raw_data["potentials"]
        u_total = np.zeros((n, n, nr))

        for i, si in enumerate(species):
            for j in range(i, n):
                sj = species[j]
                key = si + sj if si + sj in potential_dict else sj + si
                pdata = potential_dict[key]

                interp_u = interp1d(
                    pdata["r"],
                    pdata["U"],
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                )

                u_ij = beta * interp_u(r)
                u_total[i, j] = u_total[j, i] = u_ij

    # ============================================================
    # OPTIMIZATION
    # ============================================================

    pair_list = [(i, j) for i in range(n) for j in range(i, n)]
    scale_init = np.ones(len(pair_list))
    bounds = [(0.0, 10.0)] * len(scale_init)

    def objective(scale_vec):
        loss = 0.0
        k = 0
        for (i, j) in pair_list:
            f_ij = scale_vec[k]
            B2_ij = compute_B2_scaled(f_ij, r, u_total[i, j])
            diff = B2_ij - B2_target[i, j]
            loss += diff * diff
            k += 1
        return loss

    print("\nOptimizing total-potential scale factors...")

    result = minimize(
        objective,
        scale_init,
        method="Powell",
        bounds=bounds,
        options=dict(
            xtol=1e-6,
            ftol=1e-6,
            maxiter=500,
            disp=True,
        ),
    )

    scale_calibrated = unpack_scale_vector(result.x, pair_list, n)

    # ============================================================
    # COMPUTE FINAL B2 WITH CALIBRATED SCALES
    # ============================================================

    B2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            B2_ij = compute_B2_scaled(scale_calibrated[i, j], r, u_total[i, j])
            B2[i, j] = B2[j, i] = B2_ij

    # ============================================================
    # EXPORT
    # ============================================================

    if export:
        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "species": species,
                "beta": beta,
                "r_max": r_max,
                "nr": nr,
            },
            "pairs": {},
        }

        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                key = f"{si}{sj}"
                data["pairs"][key] = {
                    "B2": float(B2[i, j]),
                    "B2_target": float(B2_target[i, j]),
                    "scale_factor": float(scale_calibrated[i, j]),
                }

        path = out / f"{filename_prefix}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Second virial scale calibration exported → {path}")

    return B2, scale_calibrated

