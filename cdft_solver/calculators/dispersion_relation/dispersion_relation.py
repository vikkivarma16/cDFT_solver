import numpy as np
from pathlib import Path
import json

from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial
from cdft_solver.calculators.radial_distribution_function.rdf_two_d import rdf_2d
from cdft_solver.calculators.radial_distribution_function.rdf_radial import find_key_recursive

# -------------------------------------------------
# Dispersion relation calculator (AUTO 2D / 3D)
# -------------------------------------------------

def dispersion_relation(
    ctx,
    rdf_config,
    densities,
    diffusivity,
    supplied_data=None,
    export=False,
    plot=True,
    filename_prefix="dispersion",
):
    """
    Compute dispersion relation ω(k) using RDF input.

    Automatically selects:
        - rdf_radial (3D isotropic)
        - rdf_2d     (pure 2D)

    based on rdf_config["rdf"]["dimension"]
    """

    # -----------------------------
    # Extract config
    # -----------------------------
    rdf_block = find_key_recursive(rdf_config, "rdf")
    species = find_key_recursive(rdf_config, "species")
    r_max = rdf_block["r_max"]

    if rdf_block is None:
        raise KeyError("No 'rdf' section found")

    dimension = rdf_block.get("dimension", "3D").lower()

    densities = np.asarray(densities, float)
    diffusivity = np.asarray(diffusivity, float)

    if diffusivity.ndim == 1:
        D_matrix = np.diag(diffusivity)
    else:
        D_matrix = diffusivity

    print(f"📐 Using dimension: {dimension.upper()}")

    # ============================================================
    # STEP 1: Compute RDF
    # ============================================================

    if dimension in ["3d", "radial", "isotropic", "3D"]:
        rdf_data = rdf_radial(
            ctx=ctx,
            rdf_config=rdf_config,
            densities=densities,
            supplied_data=supplied_data,
            export=True,
            plot=False,
        )

    elif dimension in ["2d", "planar", "two_page", "2D"]:
        rdf_data = rdf_2d(
            ctx=ctx,
            rdf_config=rdf_config,
            densities=densities,
            export=True,
            plot=False,
        )

    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    # -----------------------------
    # Extract arrays
    # -----------------------------
    species = list(species)
    N = len(species)

    # assume same r-grid for all
    r = rdf_data[(species[0], species[0])]["r"]

    Nr = len(r)

    c_r = np.zeros((N, N, Nr))

    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            c_r[i, j] = rdf_data[(si, sj)]["c_r"]

    # ============================================================
    # STEP 2: Dimension-aware transform c(r) → c(k)
    # ============================================================

    def transform_matrix_dimensional(c_r, r, dimension):
        import numpy as np
        from scipy.special import j0

        dr = r[1] - r[0]
        Nr = len(r)

        # Better k-grid (physically consistent)
        r_max = r[-1]
        k = np.pi * np.arange(1, Nr + 1) / r_max

        c_k = np.zeros((N, N, Nr))

        # -----------------------------
        # 2D transform (Bessel J0)
        # -----------------------------
        if dimension.lower() in ["2d", "planar", "two_page"]:
            J0 = j0(np.outer(k, r))

            for i in range(N):
                for j in range(N):
                    c_k[i, j] = 2 * np.pi * (J0 @ (r * c_r[i, j])) * dr

        # -----------------------------
        # 3D transform (spherical)
        # -----------------------------
        elif dimension.lower() in ["3d", "radial", "isotropic"]:
            kr = np.outer(k, r)

            # safe sinc: sin(kr)/(kr)
            sinc = np.ones_like(kr)
            mask = kr != 0.0
            sinc[mask] = np.sin(kr[mask]) / kr[mask]

            for i in range(N):
                for j in range(N):
                    c_k[i, j] = 4 * np.pi * (
                        sinc @ (r**2 * c_r[i, j])
                    ) * dr

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        return c_k, k


    # Call it
    c_k, k_vals = transform_matrix_dimensional(c_r, r, dimension)

    # ============================================================
    # STEP 3: Compute dispersion relation
    # ============================================================

    Nk = len(k_vals)
    omega_k = np.zeros((Nk, N))

    for idx, k in enumerate(k_vals):

        # Thermodynamic matrix
        Ck = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                delta = 1.0 if i == j else 0.0
                Ck[i, j] = delta - densities[j] * c_k[i, j, idx]

        # Kinetic term
        K = (k**2) * D_matrix

        # Operator
        M = K @ Ck

        eigvals = np.linalg.eigvals(M)

        omega_k[idx] = -np.real(eigvals)

    dispersion_data = {
        "k": k_vals.tolist(),
        "omega": omega_k.tolist(),
    }

    # ============================================================
    # STEP 4: Output
    # ============================================================

    rdf_out = rdf_data  # already structured

    if export:
        out = Path(ctx.scratch_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_out = {
            "metadata": {
                "species": species,
                "densities": list(map(float, densities)),
                "dimension": dimension,
            },
            "pairs": {},
            "dispersion_relation": dispersion_data,
        }

        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                pair_key = f"{si}{sj}"

                json_out["pairs"][pair_key] = {
                    "r": r.tolist(),
                    "c_r": c_r[i, j].tolist(),
                }

        json_path = out / f"{filename_prefix}.json"

        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=4)

        print(f"✅ Dispersion data exported → {json_path}")

    # ============================================================
    # STEP 5: Plot
    # ============================================================

    if plot:
        import matplotlib.pyplot as plt

        plots = Path(ctx.plots_dir)
        plots.mkdir(parents=True, exist_ok=True)

        for mode in range(N):
            plt.figure()
            plt.plot(k_vals, omega_k[:, mode])
            plt.axhline(0, linestyle="--")
            plt.xlabel("k")
            plt.ylabel(f"ω_{mode}(k)")
            plt.title(f"Dispersion relation mode {mode}")
            plt.grid()

            fname = plots / f"{filename_prefix}_mode_{mode}.png"
            plt.savefig(fname, dpi=300)
            plt.close()

            print(f"Saved {fname}")

    return {
        "rdf": rdf_out,
        "dispersion": dispersion_data,
    }
