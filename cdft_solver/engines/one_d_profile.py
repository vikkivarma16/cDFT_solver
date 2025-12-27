# density functional minimizer/executor
# this is the main code which read and write the data in an executable format and then also run the program for the calculation...




def one_d_profile_executor(ctx):
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
    from cdft_solver.generators.parameters.generator_input_data_free_energy_parameters import data_exporter_free_energy_parameters as free_energy_parameters
    from cdft_solver.generators.parameters.generator_input_data_particles_interactions_parameters import data_exporter_particles_interactions_parameters as interactions_parameters
    from cdft_solver.calculators.total_free_energy.calculator_total_free_energy import total_free_energy as tfe
    from cdft_solver.calculators.total_free_energy.calculator_total_free_energy_z import total_free_energy_z as tfez
    
    from cdft_solver.generators.parameters.generator_input_data_solution_initiator import data_exporter_solution_initiator as sinit
    from cdft_solver.generators.parameters.generator_pair_potential_particles_visualization import pair_potential_particles_visualization as visualizer
    from cdft_solver.generators.parameters.generator_input_data_simulation_thermodynamic_parameters import data_exporter_simulation_thermodynamic_parameters as thermodynamic_parameters
    from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isochem import coexistence_densities_isochem as ccdi
    from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isochore import coexistence_densities_isochore as cdiso
    from cdft_solver.calculators.integrated_strength_rdf.calculator_integrated_strength_rdf import vk_rdf
    from cdft_solver.generators.parameters.generator_input_data_space_confinement_parameters import data_exporter_space_confinement_parameters as scp
    from cdft_solver.generators.parameters.generator_wall_potential_visualization import wall_potential_visualization as wpv
    from cdft_solver.generators.parameters.generator_wall_potential_values import wall_potential_values as wpva
    from cdft_solver.generators.parameters.generator_input_data_profile_parameters import data_exporter_simulation_profile_parameters as spp
    from cdft_solver.generators.grids_properties.generator_k_and_r_space_box import r_k_space_cartesian as rksc
    from cdft_solver.generators.density_weights.generator_FMT_weights_1d_cartesian import fmt_weights_1d as fw
    from cdft_solver.generators.density_weights.generator_MF_weights_1d_cartesian import mf_weights_1d as mfw
    from cdft_solver.generators.grids_properties.generator_bulk_rho_mue_r_space import bulk_rho_mue_r_space as brms
    
    from cdft_solver.calculators.one_d_profile_iterator.calculator_one_d_profile_iterator import one_d_profile_iterator_cartesian as odpic
   

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
    run_module(free_energy_parameters, [ctx], "Error exporting free energy parameters")
    run_module(tfe, [ctx], "Error computing total free energy")
    run_module(sinit, [ctx], "Error exporting solution initiator")
    run_module(visualizer, [ctx], "Error visualizing pair potentials")
    run_module(scp , [ctx], "Error exporting space confinement parameters")
    run_module(rksc, [ctx], "Error exporting grids properties")
    run_module(wpv, [ctx], "Error visualizing wall potentials")
    run_module(wpva, [ctx], "Error wall potential value evaluation")
    run_module(spp, [ctx], "Error error exporting simulation profile parameters")
    run_module(fw, [ctx], "Error exporting fmt weights")
    run_module (mfw, [ctx], "Error while computing the mean-field weights")
    
    
    # --- Step 2: Determine ensemble type ---
    ensemble = None
    json_file_solution_initiator = Path(scratch) / "input_data_solution_initiator.json"

    try:
        with open(json_file_solution_initiator, "r") as file:
            data = json.load(file)
            ensemble = data.get("ensemble", None)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Missing file: {json_file_solution_initiator}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {json_file_solution_initiator}: {e}")

    if ensemble not in ["isochores", "isochem"]:
        raise ValueError(f"❌ Unknown ensemble type '{ensemble}'. Expected 'isochores' or 'isochem'.")

    # --- Step 3: Run the appropriate coexistence density calculator ---
    if ensemble == "isochores":
        print("\n🧮 Running Isochoric Coexistence Density Calculation...\n")
        result = run_module(cdiso, [ctx], "Error computing coexistence densities (isochore mode)")
    else:
        print("\n🧮 Running Isochemical Coexistence Density Calculation...\n")
        result = run_module(ccdi, [ctx], "Error computing coexistence densities (isochem mode)")

    # --- Step 4: Export thermodynamic parameters ---
    run_module(thermodynamic_parameters, [ctx, result], "Error exporting thermodynamic parameters")

    print(f"\n ✅ Bulk executor completed successfully for ensemble '{ensemble}'.\n")
    
    print(f"\n Now density profile is being calculated.\n")
    
    
    run_module (brms, [ctx], "Error while generating bulk rho-mue on the grid")
    
    run_module (odpic, [ctx], "Error while iterating one-d Cartesian density profile")
    

    return result











