# cdft_solver/dispatchers/strength_kernel_dispatcher.py

import numpy as np
from collections.abc import Mapping
from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial


def  build_strength_kernel(
            ctx,
            config,
            supplied_data= None,
            densities= [0.1, 0.1, 0.1],
            kernel_type = "uniform",
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
    
    
   

    system_cfg = config
    
    
    
    def find_key_recursive(obj, key):
        """
        Recursively find a key in nested mappings (dict, OrderedDict, etc).
        """
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found = find_key_recursive(item, key)
                if found is not None:
                    return found
        return None
  
  
  
    species = find_key_recursive(system_cfg, "species")

    N = len (species)
    # --------------------------------------------------
    # Grid
    # --------------------------------------------------
   

    # ==================================================
    # UNIFORM KERNEL
    # ==================================================
    grid =  {}
    grid ["r_max"] = 5
    grid["n_points"] = 500
    grid["r_min"] = 5/500  
    r = np.linspace(grid["r_min"], grid["r_max"], grid["n_points"], endpoint=True )

    Nr = grid["n_points"]
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
    system = config

    if kernel_type == "rdf":
        print("🔄 Computing RDF-based integrated strength kernel for densities:", densities)

        # Compute RDF output
        rdf_out = rdf_radial(
            ctx=ctx,
            rdf_config=config,
            densities=densities,
            supplied_data=None,
            export=False,
            plot=True,
            filename_prefix="rdf",
        )

        # Get species list
        species = find_key_recursive(system_cfg, "species")
        N = len(species)

        # Infer r-grid size from one of the RDF outputs
        sample_key = (species[0], species[0])
        r = rdf_out[sample_key]["r"]
        Nr = len(r)

        # Initialize kernel array
        kernel = np.zeros((N, N, Nr))

        # Fill kernel from RDF
        for i, si in enumerate(species):
            for j, sj in enumerate(species):
                kernel[i, j, :] = rdf_out[(si, sj)]["g_r"]

        return {
            "type": "rdf",
            "kernel": kernel,
            "r": r,
        }


    # ==================================================
    # ERROR
    # ==================================================
    raise ValueError(
        f"Unknown integrated_strength_kernel: '{kernel_type}' "
        "(expected 'uniform' or 'rdf')"
    )

