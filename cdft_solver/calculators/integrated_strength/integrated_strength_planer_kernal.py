import numpy as np
import json
from pathlib import Path
from collections.abc import Mapping


def vij_planer_kernel(
    ctx,
    config,
    kernel_data,
    u_data,
    export_json=False,
    filename="vij_planar_kernel_u.json",
    plot=False,
):
    """
    Compute planar MF coupling:

        v_ij(z1,z2) = ∫ 2π r K_ij(z1,z2,r) U_ij(z1,z2,r) dr

    Kernel and U must both be (Nz, Nz, Nr).
    """

    # --------------------------------------------------
    # Recursive lookup
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

    species = find_key_recursive(config, "species")
    if species is None:
        raise ValueError("Species not found in config")

    # --------------------------------------------------
    # Grids
    # --------------------------------------------------
    z_grid = np.asarray(kernel_data["z_grid"], dtype=float)
    r_grid = np.asarray(kernel_data["r_grid"], dtype=float)

    Nz = len(z_grid)
    Nr = len(r_grid)

    K_dict = kernel_data["strength_kernel"]
    U_dict = u_data["mf_weights"]

    vij_numeric = {}

    # Precompute radial prefactor
    radial_weight = 2.0 * np.pi * r_grid

    # --------------------------------------------------
    # Integration
    # --------------------------------------------------
    for i, si in enumerate(species):
        for j, sj in enumerate(species):

            pair = f"{si}{sj}"
            rpair = f"{sj}{si}"

            # --- fetch kernel ---
            if pair in K_dict:
                K = np.asarray(K_dict[pair], dtype=float)
            elif rpair in K_dict:
                K = np.asarray(K_dict[rpair], dtype=float)
            else:
                raise KeyError(f"Missing strength kernel for pair {si}-{sj}")

            # --- fetch MF weight ---
            if pair in U_dict:
                U = np.asarray(U_dict[pair], dtype=float)
            elif rpair in U_dict:
                U = np.asarray(U_dict[rpair], dtype=float)
            else:
                raise KeyError(f"Missing MF weight for pair {si}-{sj}")

            if K.shape != (Nz, Nz, Nr):
                raise ValueError(
                    f"Kernel shape mismatch for {pair}: {K.shape} != {(Nz, Nz, Nr)}"
                )

            if U.shape != (Nz, Nz, Nr):
                raise ValueError(
                    f"MF weight shape mismatch for {pair}: {U.shape} != {(Nz, Nz, Nr)}"
                )

            # --------------------------------------------------
            # Vectorized radial integration
            # --------------------------------------------------
            integrand = radial_weight[None, None, :] * K * U

            vij = np.trapz(integrand, r_grid, axis=2)

            vij_numeric[(si, sj)] = vij

            print(f"✅ vij(z,z′) computed for {si}-{sj}")

    # --------------------------------------------------
    # Export JSON
    # --------------------------------------------------
    if export_json:
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)

        out = scratch / filename

        export_dict = {
            "species": species,
            "z_grid": z_grid.tolist(),
            "vij": {},
        }

        for (si, sj), vij in vij_numeric.items():
            if species.index(si) <= species.index(sj):
                export_dict["vij"][f"{si}_{sj}"] = vij.tolist()

        with open(out, "w") as f:
            json.dump(export_dict, f, indent=2)

        print(f"✅ vij matrix exported → {out}")

    # --------------------------------------------------
    # Optional plotting
    # --------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt

        plot_dir = (
            Path(ctx.plots_dir)
            if hasattr(ctx, "plots_dir")
            else Path(ctx.scratch_dir) / "plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)

        for (si, sj), vij in vij_numeric.items():

            # Only plot upper triangle once
            if species.index(si) > species.index(sj):
                continue

            z_indices = np.linspace(0, Nz - 1, min(10, Nz), dtype=int)

            plt.figure(figsize=(7, 5))

            print(f"\n🔎 Integrated vij(z) for pair {si}-{sj}:")

            for idx in z_indices:
                z0 = z_grid[idx]
                delta_z = z0 - z_grid

                plt.plot(
                    delta_z,
                    vij[idx, :],
                    lw=1.2,
                    label=f"z={z0:.3f}"
                )

                vij_int = np.trapz(vij[idx, :], z_grid)

                print(f"   z = {z0: .5f}  →  ∫dz' v_ij = {vij_int: .6e}")

            plt.axvline(0.0, color="k", ls="--", lw=0.8)
            plt.xlabel(r"$z - z'$")
            plt.ylabel(rf"$v_{{{si}{sj}}}(z,z')$")
            plt.title(f"Planar MF coupling: {si}-{sj}")
            plt.legend(fontsize=8, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            figfile = plot_dir / f"vij_planar_{si}_{sj}.png"
            plt.savefig(figfile, dpi=300)
            plt.close()

            print(f"📊 vij(z−z′) plot saved → {figfile}")

    return {
        "species": species,
        "z_grid": z_grid,
        "vij_numeric": vij_numeric,
    }
