import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


def mf_weights_two_d_cylindrical(ctx, plot=False):
    """
    Build MF weights in real space for cylindrical symmetry.

    Uses supplied_data_r_space.txt with columns:
        z, r, phi   (phi ignored)

    For each pair ij:
        U(z,z'; r) = U_iso( sqrt((z-z')^2 + r^2) )

    Output:
        supplied_data_weight_mf_z_zp_ij.txt
    """

    scratch = Path(ctx.scratch_dir)

    # --------------------------------------------------
    # Load interaction data → species & pair keys
    # --------------------------------------------------
    json_path = scratch / "input_data_particles_interactions_parameters.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing interaction file: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    species = data["particles_interactions_parameters"]["species"]
    interactions = data["particles_interactions_parameters"]["interactions"]

    levels = ["primary", "secondary", "tertiary"]

    pair_keys = set()
    for level in levels:
        for key in interactions.get(level, {}):
            pair_keys.add(key)

    if not pair_keys:
        print("⚠️ No interaction pairs found — nothing to do.")
        return

    # --------------------------------------------------
    # Load cylindrical r-space: (z, r, phi)
    # --------------------------------------------------
    rspace_file = scratch / "supplied_data_r_space.txt"
    if not rspace_file.exists():
        raise FileNotFoundError("Missing supplied_data_r_space.txt")

    rspace = np.loadtxt(rspace_file)

    if rspace.shape[1] < 2:
        raise ValueError("r-space file must contain at least z and r columns")

    z_all = rspace[:, 0]
    r_all = rspace[:, 1]

    # Unique sorted grids
    z = np.unique(z_all)
    r_parallel = np.unique(r_all)

    Nz = len(z)
    Nr = len(r_parallel)

    # --------------------------------------------------
    # Loop over all pairs
    # --------------------------------------------------
    for key in sorted(pair_keys):

        pot_file = scratch / f"supplied_data_potential_mf_{key}.txt"
        if not pot_file.exists():
            print(f"⚠️ Skipping pair {key}: MF potential not found")
            continue

        pot_data = np.loadtxt(pot_file)
        R_grid = pot_data[:, 0]
        U_R = pot_data[:, 1]
        R_cut = R_grid.max()

        # Interpolator for isotropic MF potential
        U_interp = interp1d(
            R_grid,
            U_R,
            bounds_error=False,
            fill_value=0.0
        )

        out_file = scratch / f"supplied_data_weight_mf_z_zp_{key}.txt"
        with open(out_file, "w") as fout:
            fout.write("# z   z'   r_parallel   U(z,z'; r_parallel)\n")

            for r in r_parallel:
                for zi in z:
                    for zj in z:
                        R = np.sqrt((zi - zj) ** 2 + r ** 2)
                        Uval = 0.0 if R > R_cut else float(U_interp(R))
                        fout.write(
                            f"{zi:.8e} {zj:.8e} {r:.8e} {Uval:.8e}\n"
                        )
                fout.write("\n")

        print(f"✅ MF z–z′ kernel written: {out_file}")

        # --------------------------------------------------
        # Optional diagnostic plot (saved)
        # --------------------------------------------------
        if plot:
            import matplotlib.pyplot as plt

            r0 = r_parallel[0]
            Umat = np.zeros((Nz, Nz))

            for i, zi in enumerate(z):
                for j, zj in enumerate(z):
                    R = np.sqrt((zi - zj) ** 2 + r0 ** 2)
                    Umat[i, j] = 0.0 if R > R_cut else U_interp(R)

            plt.figure(figsize=(6, 5))
            plt.imshow(
                Umat,
                extent=[z.min(), z.max(), z.min(), z.max()],
                origin="lower",
                aspect="auto"
            )
            plt.colorbar(label="U(z,z')")
            plt.xlabel("z")
            plt.ylabel("z'")
            plt.title(f"MF kernel for pair {key} (r={r0:.3f})")
            plt.tight_layout()

            figfile = scratch / f"vis_weight_mf_z_zp_{key}.png"
            plt.savefig(figfile, dpi=300)
            plt.close()

            print(f"📊 Plot saved: {figfile}")

    print("\n✅ All MF cylindrical real-space weights generated successfully.\n")

