# cdft_solver/dispatchers/strength_kernel_dispatcher.py

import numpy as np

from cdft_solver.generators.rdf_isotropic import rdf_isotropic


def build_strength_kernel(
    ctx,
    config,
    grid_dict,
    potential_dict,
    densities,
    sigma_matrix=None,
    supplied_data=None,
):
    """
    Dispatcher for integrated strength kernel.

    Density is ALWAYS uniform.
    Decision is based ONLY on:
        system.integrated_strength_kernel

    Parameters
    ----------
    ctx : object
        Context with scratch_dir / plots_dir
    config : dict
        Full configuration dictionary containing "system" block
    grid_dict : dict
        {
            "r_min": float,
            "r_max": float,
            "n_points": int
        }
    potential_dict : dict
        Pair potential dictionary
    densities : array-like
        Species densities
    sigma_matrix : ndarray, optional
        Hard-core diameters
    supplied_data : dict, optional
        Supplied RDF constraints (used only if enabled)

    Returns
    -------
    dict
        {
            "type": "uniform" | "rdf",
            "kernel": ndarray (N, N, Nr),
            "r": ndarray or None,
            "rdf_raw": dict (only if rdf)
        }
    """

    system_cfg = config.get("system", {})
    kernel_type = system_cfg.get("integrated_strength_kernel", "uniform")
    supplied_flag = system_cfg.get("supplied_data", "no").lower() == "yes"

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
    r = np.linspace(
        grid_dict["r_min"],
        grid_dict["r_max"],
        grid_dict["n_points"],
    )

    Nr = len(r)
    N = len(densities)

    # ==================================================
    # UNIFORM KERNEL
    # ==================================================
    if kernel_type == "uniform":
        print("✅ Using UNIFORM integrated strength kernel")

        kernel = np.ones((N, N, Nr))

        return {
            "type": "uniform",
            "kernel": kernel,
            "r": r,
        }

    # ==================================================
    # RDF KERNEL
    # ==================================================
    if kernel_type == "rdf":
        print("🔄 Computing RDF-based integrated strength kernel")

        rdf_out = rdf_isotropic(
            ctx=ctx,
            rdf_config=config,
            grid_dict=grid_dict,
            potential_dict=potential_dict,
            densities=densities,
            sigma=sigma_matrix,
            supplied_data=supplied_data if supplied_flag else None,
            export=config.get("export_rdf", False),
            plot=config.get("plot_rdf", False),
        )

        species = config["rdf_parameters"]["species"]
        Nr = len(r)

        kernel = np.zeros((N, N, Nr))

        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                kernel[i, j, :] = rdf_out[(si, sj)]["g_r"]

        return {
            "type": "rdf",
            "kernel": kernel,
            "r": r,
            "rdf_raw": rdf_out,
        }

    # ==================================================
    # ERROR
    # ==================================================
    raise ValueError(
        f"Unknown integrated_strength_kernel: '{kernel_type}' "
        "(expected 'uniform' or 'rdf')"
    )

