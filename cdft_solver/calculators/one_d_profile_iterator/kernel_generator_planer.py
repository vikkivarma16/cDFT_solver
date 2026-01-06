# cdft_solver/dispatchers/strength_kernel_dispatcher.py

import numpy as np
from collections.abc import Mapping
from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial


def build_strength_kernel_planer(
    ctx,
    config,
    densities,
    supplied_data=None,
    kernel_type="uniform",
):
    """
    Build planar strength kernel with MF-compatible structure.

    Returns
    -------
    dict:
        {
            "species": [...],
            "z_grid": [...],
            "r_grid": [...],
            "strength_kernel": {
                "AA": (Nz, Nz, Nr),
                "AB": (Nz, Nz, Nr),
                ...
            }
        }
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

    # --------------------------------------------------
    # Species
    # --------------------------------------------------
    species = find_key_recursive(config, "species")
    if species is None:
        raise ValueError("Species not found in config")

    N = len(species)

    # --------------------------------------------------
    # Planar grid
    # --------------------------------------------------
    params = find_key_recursive(config, "planer_rdf")
    if params is None:
        raise ValueError("Missing planer_rdf block")

    box_props = params["box_properties"]
    Nz = int(box_props["box_points"][0])
    Nr = int(box_props["box_points"][1])

    Lz = float(box_props["box_length"][0])
    Rmax = float(box_props["box_length"][1])

    z_grid = np.linspace(0.0, Lz, Nz)
    r_grid = np.linspace(0.0, Rmax, Nr)

    # --------------------------------------------------
    # Allocate kernel
    # --------------------------------------------------
    strength_kernel = {}

    def pair_key(a, b):
        return f"{a}{b}"

    # ==================================================
    # UNIFORM KERNEL
    # ==================================================
    if kernel_type == "uniform":
        print("✅ Using UNIFORM planar strength kernel")

        for i, si in enumerate(species):
            for j, sj in enumerate(species[i:], start=i):
                pair = pair_key(si, sj)
                kernel = np.ones((Nz, Nz, Nr), dtype=float)
                strength_kernel[pair] = kernel
                strength_kernel[pair_key(sj, si)] = kernel

        return {
            "type": "uniform",
            "species": species,
            "z_grid": z_grid.tolist(),
            "r_grid": r_grid.tolist(),
            "strength_kernel": {
                k: v.tolist() for k, v in strength_kernel.items()
            },
        }

    # ==================================================
    # RDF KERNEL
    # ==================================================
    if kernel_type == "rdf":
        print("🔄 Computing RDF-based planar strength kernel")

        rdf_out = rdf_radial(
            ctx=ctx,
            rdf_config=config,
            densities=densities,
            supplied_data=supplied_data,
            export=False,
            plot=False,
            filename_prefix="rdf",
        )

        for i, si in enumerate(species):
            for j, sj in enumerate(species[i:], start=i):

                g_r = rdf_out[(si, sj)]["g_r"]
                r_rdf = rdf_out[(si, sj)]["r"]

                if len(g_r) != Nr:
                    raise ValueError("RDF r-grid incompatible with planar grid")

                kernel = np.zeros((Nz, Nz, Nr), dtype=float)

                # Planar lift: same g(r) for every (z,z′)
                for k in range(Nr):
                    kernel[:, :, k] = g_r[k]

                strength_kernel[pair_key(si, sj)] = kernel
                strength_kernel[pair_key(sj, si)] = kernel

        return {
            "type": "rdf",
            "species": species,
            "z_grid": z_grid.tolist(),
            "r_grid": r_grid.tolist(),
            "strength_kernel": {
                k: v.tolist() for k, v in strength_kernel.items()
            },
        }

    # ==================================================
    # ERROR
    # ==================================================
    raise ValueError(
        f"Unknown kernel_type '{kernel_type}' (expected 'uniform' or 'rdf')"
    )

