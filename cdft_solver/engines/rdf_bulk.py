def rdf_bulk_executor(ctx):
    """
    Bulk executor for Density Functional Theory simulations.
    Automatically determines the ensemble type (isochore or isochem)
    and runs the corresponding coexistence density calculator.
    """

    import json
    from pathlib import Path

    # --- Scientific imports ---
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import integrate
    from scipy.special import j0
    import pyfftw.interfaces as fftw
    import sympy as sp
    from sympy import log, diff, lambdify

    # --- cdft_solver modules ---
    
    from cdft_solver.generators.parameters.generator_input_data_particles_interactions_parameters import data_exporter_particles_interactions_parameters as interactions_parameters
    from cdft_solver.generators.parameters.generator_pair_potential_particles_visualization import pair_potential_particles_visualization as visualizer
    from cdft_solver.generators.parameters.generator_input_data_iso_rdf_parameters import data_exporter_iso_rdf_parameters as irp
    from cdft_solver.calculators.radial_distribution_function.calculator_rdf_isotropic import rdf_isotropic as ri

    # --- Paths ---
    scratch = Path(ctx.scratch_dir)
    plots = Path(ctx.plots_dir)
    input_file = Path(ctx.input_file)

    def run_module(module_func, args=None, err_msg="Error running module"):
        """Safely run a module with arguments and error handling."""
        try:
            if args is None:
                return module_func()
            elif isinstance(args, (list, tuple)):
                return module_func(*args)
            else:
                return module_func(args)
        except Exception as e:
            print(f"{err_msg}: {e}")
            exit(1)

    # --- Step 1: Generate required input data ---
    run_module(interactions_parameters, [ctx], "Error exporting particle interactions")
    run_module(visualizer, [ctx], "Error visualizing pair potentials")
    
    print
    run_module(irp, [ctx], "Error exporting rdf parameters")
    run_module(ri, [ctx], "Error computing rdf")
    
   

    print(f"\n✅ Bulk executor completed successfully for rdf computation.\n")

    return 0

