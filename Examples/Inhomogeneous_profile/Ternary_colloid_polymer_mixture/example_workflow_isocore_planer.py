from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
from cdft_solver.calculators.one_d_profile_iterator.one_d_profile_iterator_box import one_d_profile_iterator_box




# define your directory to export the data and the plots

scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)


# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_isocore.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system =  super_dictionary_creator (ctx, export_json =  True, filename  = "input_system.json", super_key_name =  "system")





one_d_profile_iterator_box (ctx = ctx, config = system, export_json= True, export_plots = True, filename  = "one_d_profiles")


exit(0)
