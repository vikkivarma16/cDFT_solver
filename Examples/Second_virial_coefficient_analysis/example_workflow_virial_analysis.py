import numpy as np
from pathlib import Path
from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
from cdft_solver.generators.supplied_data.process_supplied_data import process_supplied_data as psd
from cdft_solver.calculators.virial_analysis.second_virial import second_virial
from cdft_solver.calculators.virial_analysis.second_virial_scale_calibration import second_virial_scale_calibration





# define your directory to export the data and the plots

scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)


# export different dictionaries bases on their functions.
ctx_ref = ExecutionContext(
    input_file="example_input_virial_analysis.in",
    scratch_dir=scratch,
    plots_dir=plots,
)




system_ref =  super_dictionary_creator (ctx = ctx_ref, export_json = True, filename = "input_system.json", super_key_name = "system")

supplied_data = psd( ctx = ctx_ref, config = system_ref, export_json = True, export_plot = True)

b2t, strength  = second_virial(ctx = ctx_ref, virial_config = system_ref, on = "raw",  export=True, filename_prefix="second_virial_coefficient", r_max_factor=12.0, nr=8192, n_lambda=128, )




ctx_target = ExecutionContext(
    input_file="example_input_virial_analysis_calibration.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system_target = super_dictionary_creator (ctx =  ctx_target, export_json = True, filename = "input_system_target.json", super_key_name = "system")

print (b2t)


b2t = [[-0.125]]

result_new = second_virial_scale_calibration(ctx = ctx_target, virial_config = system_target, B2_target = b2t, on="raw", export=True, filename_prefix="second_virial_coefficient", r_max_factor=12.0, nr=8192, beta_scale=1.0,)

print (result_new)




