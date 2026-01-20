import numpy as np
from pathlib import Path
from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
from cdft_solver.calculators.radial_distribution_function.rdf_planer import rdf_planer
from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial
from cdft_solver.generators.supplied_data.process_supplied_data import process_supplied_data as psd
from cdft_solver.calculators.rdf_inversion.rdf_inversion_standard import boltzmann_inversion_standard





# define your directory to export the data and the plots

scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)


# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_inversion.in",
    scratch_dir=scratch,
    plots_dir=plots,
)




system =  super_dictionary_creator (ctx, export_json = True, filename = "input_system.json", super_key_name = "system")

supplied_data = psd( ctx = ctx, config = system , export_json = True, export_plot = True)

boltzmann_inversion_standard( ctx = ctx, rdf_config= system, supplied_data =  supplied_data, filename_prefix="multistate", export_plot=True, export_json=True )

