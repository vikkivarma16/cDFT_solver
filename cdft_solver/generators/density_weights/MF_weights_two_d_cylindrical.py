# cdft_solver/mf/mf_weights_two_d_cylindrical.py

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials as mmf

def mf_weights_two_d_cylindrical(
    ctx=None,
    data_dict=None,
    grid_properties=None,
    export_json=True,
    filename="supplied_data_weight_mf_cylindrical.json",
    plot=False
):
    """
    Dictionary-driven MF cylindrical kernel generator.

    Everything (species, grids, potentials) is supplied via data_dict.

    Builds:
        U(z, z'; r) = U_iso( sqrt((z-z')^2 + r^2) )

    Exports per-pair potential matrices.
    """

    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx with scratch_dir must be provided")

    if data_dict is None:
        raise ValueError("data_dict must be provided")

    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = scratch_dir / "plots"
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Recursive helpers
    # --------------------------------------------------
    def find_key_recursive(d, key):
        if not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        return None

    # --------------------------------------------------
    # Extract required data
    # --------------------------------------------------
    species = find_key_recursive(data_dict, "species")
    if not species:
        raise KeyError("No 'species' found in dictionary")
    
    
    mf_data = mmf( ctx=ctx, input_data=data_dict, grid_points=5000, file_name_prefix="supplied_data_potential_mf.json", export_files=False )
    mf_potentials = mf_data["potentials"]
    
    if not mf_potentials:
        raise KeyError("No 'mf_potentials' found in dictionary")

   # Extract r-space
    r_space = np.array(grid_properties["r_space"]) 

    r_space = np.array(r_space, dtype=float)
    if r_space.shape[1] < 2:
        raise ValueError("r_space must have at least 2 columns (z, r)")

    z_all = r_space[:, 0]
    r_all = r_space[:, 1]

    # Unique sorted grids
    z = np.unique(z_all)
    r_parallel = np.unique(r_all)
    Nz = len(z)
    Nr = len(r_parallel)


    # --------------------------------------------------
    # Build all species pairs
    # --------------------------------------------------
    def make_pair(a, b):
        return f"{a}{b}"

    pairs = []
    for i, si in enumerate(species):
        for sj in species[i:]:
            pairs.append(make_pair(si, sj))

    # --------------------------------------------------
    # Compute MF kernels
    # --------------------------------------------------
    result = {
        "species": species,
        "z_grid": z.tolist(),
        "r_grid": r_parallel.tolist(),
        "mf_weights": {}
    }

    for pair in pairs:

        if pair not in mf_potentials:
            print(f"⚠️ Skipping {pair}: no potential supplied")
            continue

        pot_data = mf_potentials[pair]
        R_grid = np.array(pot_data["R"], dtype=float)
        U_R = np.array(pot_data["U"], dtype=float)

        R_cut = R_grid.max()
        U_interp = interp1d(R_grid, U_R, bounds_error=False, fill_value=0.0)

        Umat = np.zeros((Nz, Nz, Nr))

        for k, r in enumerate(r_parallel):
            for i, zi in enumerate(z):
                for j, zj in enumerate(z):
                    R = np.sqrt((zi - zj)**2 + r**2)
                    Umat[i, j, k] = 0.0 if R > R_cut else float(U_interp(R))

        result["mf_weights"][pair] = Umat.tolist()

        print(f"✅ MF kernel built for pair {pair}")

        # --------------------------------------------------
        # Optional plot (z–z′ slice at r[0])
        # --------------------------------------------------
        if plot:
            plt.figure(figsize=(6, 5))
            plt.imshow(
                Umat[:, :, 0],
                extent=[z.min(), z.max(), z.min(), z.max()],
                origin="lower",
                aspect="auto"
            )
            plt.colorbar(label="U(z,z')")
            plt.xlabel("z")
            plt.ylabel("z'")
            plt.title(f"MF kernel {pair} (r={r_parallel[0]:.3f})")
            plt.tight_layout()

            figfile = plot_dir / f"mf_weight_{pair}.png"
            plt.savefig(figfile, dpi=300)
            plt.close()

            print(f"📊 Plot saved: {figfile}")

    # --------------------------------------------------
    # Export JSON
    # --------------------------------------------------
    if export_json:
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ MF cylindrical weights exported to {out_file}")

    return result

