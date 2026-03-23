import numpy as np
import sympy as sp
import json
from pathlib import Path
from scipy.interpolate import interp1d
from collections.abc import Mapping

from cdft_solver.generators.potential_splitter.mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials


def vij_radial_kernel(
    ctx,
    config,
    kernel,
    supplied_data=None,
    export_json=False,
    filename="vij_integrated.json",
):
    """
    Compute:
        v_ij = ∫ 4π r² K_ij(r) U_ij(r) dr

    Strategy:
    ----------
    If:
        - kernel ~ uniform
        - AND hard-core present
    then:
        v_ij = 2 ΔB2 (computed from RAW potential)
    """

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def find_key_recursive(obj, key):
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
            for v in obj.values():
                out = find_key_recursive(v, key)
                if out is not None:
                    return out
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out = find_key_recursive(v, key)
                if out is not None:
                    return out
        return None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def is_uniform_kernel(K, tol=1e-3):
        return np.std(K) < tol and abs(np.mean(K) - 1.0) < tol

    def has_hard_core(U, threshold=500.0):
        return np.max(U) > threshold

    def compute_B2(r, u_r, beta):
        f_r = np.exp(-beta * u_r) - 1.0
        return -2.0 * np.pi * np.trapz(r**2 * f_r, r)

    def wca_split(r, u):
        idx_min = np.argmin(u)
        u_min = u[idx_min]

        u_rep = np.zeros_like(u)
        mask = r <= r[idx_min]
        u_rep[mask] = u[mask] - u_min
        return u_rep

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    species = find_key_recursive(config, "species")
    beta = find_key_recursive(config, "beta")

    if species is None:
        raise ValueError("Species list not found in config")
    if beta is None:
        raise ValueError("Beta not found in config")

    n_species = len(species)
    n_grid = 5000

    # --------------------------------------------------
    # Symbolic matrix
    # --------------------------------------------------
    vij_symbols = [
        [sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
        for i in range(n_species)
    ]

    vij_numeric = {}

    # --------------------------------------------------
    # Load potentials
    # --------------------------------------------------
    mf_data = meanfield_potentials(
        ctx=ctx,
        input_data=config,
        grid_points=n_grid,
        export_files=False,
    )

    raw_data = raw_potentials(
        ctx=ctx,
        input_data=config,
        grid_points=n_grid,
        export_files=False,
    )

    hc_data = hard_core_potentials(
        ctx=ctx,
        input_data=config,
        grid_points=n_grid,
        export_files=False,
    )

    sigma_matrix = np.array(hc_data.get("sigma", None))

    mf_dict = mf_data["potentials"]
    raw_dict = raw_data["potentials"]

    # --------------------------------------------------
    # Build dictionaries
    # --------------------------------------------------
    U_dict = {}
    Uraw_dict = {}

    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):

            key_ij = si + sj
            key_ji = sj + si

            # --- MF
            pdata = mf_dict.get(key_ij) or mf_dict.get(key_ji)
            if pdata is None:
                raise KeyError(f"Missing MF potential for {si}-{sj}")

            r = np.asarray(pdata["r"], float)
            U = beta * np.asarray(pdata["U"], float)

            U_dict[(si, sj)] = {"r": r, "U": U}
            U_dict[(sj, si)] = {"r": r, "U": U}

            # --- RAW
            pdata_raw = raw_dict.get(key_ij) or raw_dict.get(key_ji)
            if pdata_raw is None:
                raise KeyError(f"Missing RAW potential for {si}-{sj}")

            r_raw = np.asarray(pdata_raw["r"], float)
            U_raw = np.asarray(pdata_raw["U"], float)

            Uraw_dict[(si, sj)] = {"r": r_raw, "U": U_raw}
            Uraw_dict[(sj, si)] = {"r": r_raw, "U": U_raw}

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):

            key = (si, sj)
            rkey = (sj, si)

            # kernel
            ker = kernel.get(key) or kernel.get(rkey)
            if ker is None:
                raise KeyError(f"Missing kernel for {si}-{sj}")

            rk = np.asarray(ker["r"], float)
            K = np.asarray(ker["values"], float)

            # MF potential
            ru = U_dict[key]["r"]
            Uv = U_dict[key]["U"]

            # RAW potential
            rr = Uraw_dict[key]["r"]
            Ur = Uraw_dict[key]["U"]

            # overlap
            r_lo = max(rk.min(), ru.min(), rr.min())
            r_hi = min(rk.max(), ru.max(), rr.max())

            if r_hi <= r_lo:
                raise ValueError(f"No overlapping domain for {si}-{sj}")

            r_common = np.linspace(r_lo, r_hi, n_grid)

            Kc = interp1d(rk, K, bounds_error=False, fill_value=0.0)(r_common)
            Uc = interp1d(ru, Uv, bounds_error=False, fill_value=0.0)(r_common)
            Uc_raw = interp1d(rr, Ur, bounds_error=False, fill_value=0.0)(r_common)

            # --------------------------------------------------
            # Decision
            # --------------------------------------------------
            use_b2 = is_uniform_kernel(Kc) and has_hard_core(Uc_raw)

            if use_b2:
                import matplotlib.pyplot as plt

                sigma = None
                if sigma_matrix is not None:
                    sigma = sigma_matrix[i, j]

                u_real = Uc_raw
                u_ref = wca_split(r_common, u_real)

                print(u_real)

                # --------------------------------------------------
                # B2 computation
                # --------------------------------------------------
                B2_real = compute_B2(r_common, u_real, beta)
                B2_ref = compute_B2(r_common, u_ref, beta)

                vij = 2.0 * (B2_real - B2_ref)

                print(beta)
                print(f"[DEBUG] Pair ({si},{sj})")
                print(f"B2_real = {B2_real:.6e}, B2_ref = {B2_ref:.6e}")
                print(f"vij (2ΔB2) = {vij:.6e}")

                # --------------------------------------------------
                # Prepare debug data
                # --------------------------------------------------
                exp_real = np.exp(-beta * u_real)
                exp_ref  = np.exp(-beta * u_ref)

                scratch = Path(ctx.scratch_dir)
                scratch.mkdir(parents=True, exist_ok=True)

                # --------------------------------------------------
                # SAVE NPZ (full precision)
                # --------------------------------------------------
                npz_file = scratch / f"debug_b2_data_{si}_{sj}.npz"
                np.savez(
                    npz_file,
                    r=r_common,
                    u_real=u_real,
                    u_ref=u_ref,
                    exp_real=exp_real,
                    exp_ref=exp_ref,
                    beta=beta,
                    B2_real=B2_real,
                    B2_ref=B2_ref,
                    vij=vij,
                )
                print(f"[DEBUG] NPZ saved → {npz_file}")

                # --------------------------------------------------
                # SAVE CLEAN TXT (readable)
                # --------------------------------------------------
                txt_file = scratch / f"debug_b2_data_{si}_{sj}.txt"

                with open(txt_file, "w") as f:
                    f.write("# ==========================================\n")
                    f.write(f"# Pair: {si}-{sj}\n")
                    f.write(f"# beta = {beta}\n")
                    f.write(f"# B2_real = {B2_real:.8e}\n")
                    f.write(f"# B2_ref  = {B2_ref:.8e}\n")
                    f.write(f"# vij (2ΔB2) = {vij:.8e}\n")
                    if sigma is not None:
                        f.write(f"# sigma = {sigma}\n")
                    f.write("# ==========================================\n")
                    f.write("# Columns:\n")
                    f.write("# r        u_real        u_ref        exp(-βu_real)    exp(-βu_ref)\n")

                    for rr, ur, uref, er, eref in zip(
                        r_common, u_real, u_ref, exp_real, exp_ref
                    ):
                        f.write(
                            f"{rr:12.6f}  {ur:14.6e}  {uref:14.6e}  {er:14.6e}  {eref:14.6e}\n"
                        )

                print(f"[DEBUG] TXT saved → {txt_file}")

                # --------------------------------------------------
                # SAVE CSV (Excel-friendly)
                # --------------------------------------------------
                csv_file = scratch / f"debug_b2_data_{si}_{sj}.csv"

                with open(csv_file, "w") as f:
                    f.write("r,u_real,u_ref,exp_real,exp_ref\n")
                    for rr, ur, uref, er, eref in zip(
                        r_common, u_real, u_ref, exp_real, exp_ref
                    ):
                        f.write(f"{rr},{ur},{uref},{er},{eref}\n")

                print(f"[DEBUG] CSV saved → {csv_file}")

                # --------------------------------------------------
                # Plot potentials
                # --------------------------------------------------
                fig = plt.figure()

                plt.plot(r_common, u_real, label="u_real (raw)")
                plt.plot(r_common, u_ref, "--", label="u_ref (WCA)")

                if sigma is not None and sigma > 0:
                    plt.axvline(sigma, linestyle=":", label=f"sigma = {sigma:.3f}")

                plt.xlabel("r")
                plt.ylabel("u(r)")
                plt.title(f"Pair {si}-{sj}\n2ΔB2 = {vij:.4e}")
                plt.legend()
                plt.ylim(-1, 1)
                plt.grid(True)

                fname = scratch / f"debug_u_real_ref_{si}_{sj}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

                print(f"[DEBUG] Plot saved → {fname}")

                # --------------------------------------------------
                # Plot integrand (CRITICAL DEBUG)
                # --------------------------------------------------
                fig2 = plt.figure()

                integrand_real = (exp_real - 1.0) * r_common**2
                integrand_ref  = (exp_ref - 1.0) * r_common**2

                plt.plot(r_common, integrand_real, label="integrand real")
                plt.plot(r_common, integrand_ref, "--", label="integrand ref")

                plt.xlabel("r")
                plt.ylabel("r^2 (exp(-βu)-1)")
                plt.title(f"B2 Integrand {si}-{sj}")
                plt.legend()
                plt.grid(True)

                fname2 = scratch / f"debug_integrand_{si}_{sj}.png"
                plt.savefig(fname2, dpi=150, bbox_inches="tight")
                plt.close(fig2)

                print(f"[DEBUG] Integrand plot saved → {fname2}")

                # optional hard stop
                exit(0)

            else:
                vij = float(
                    np.trapz(4.0 * np.pi * r_common**2 * Kc * Uc, r_common)
                )

            vij_numeric[key] = vij
            vij_numeric[rkey] = vij

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    if export_json:
        out = scratch / filename
        with open(out, "w") as f:
            json.dump(
                {
                    "species": species,
                    "vij": {
                        f"{k[0]}_{k[1]}": v for k, v in vij_numeric.items()
                    },
                },
                f,
                indent=4,
            )

    return {
        "species": species,
        "vij_symbols": vij_symbols,
        "vij_numeric": vij_numeric,
    }
