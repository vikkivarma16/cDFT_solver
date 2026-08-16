from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
from cdft_solver.calculators.radial_distribution_function.rdf_planer import rdf_planer
from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial
import numpy as np



# define your directory to export the data and the plots

scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)


# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_rdf_radial.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system =  super_dictionary_creator (ctx, export_json =  True, filename  = "input_system.json", super_key_name =  "system")




rho = [0.3]


output  = rdf_radial(ctx, rdf_config = system, densities = rho, supplied_data=None, export=True, plot=True, filename_prefix="rdf")
