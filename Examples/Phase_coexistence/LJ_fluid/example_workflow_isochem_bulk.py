from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials
from cdft_solver.calculators.total_free_energy.total_free_energy import total_free_energy
from cdft_solver.calculators.total_free_energy.free_energy_exporter  import free_energy_exporter

from cdft_solver.calculators.radial_distribution_function.rdf_radial import rdf_radial
from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isochem import coexistence_densities_isochem



# define your directory to export the data and the plots



scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)




# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_isochem.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system =  super_dictionary_creator (ctx, export_json =  True, filename  = "input_system.json", super_key_name =  "system")




#exit (0)

hc_data = hard_core_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_hc.json",
    export_files=True
)


mf_data = meanfield_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_mf.json",
    export_files=True
)


raw_data = raw_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_raw.json",
    export_files=True
)


total_data = total_potentials(
    ctx=ctx,
    hc_source= hc_data,
    mf_source= mf_data,
    file_name_prefix="supplied_data_potential_total.json",
    export_files=True,
   
)

filenames = {}
filenames["hard_core"] = "supplied_data_free_energy_hard_core.json"
filenames["mean_field"] = "supplied_data_free_energy_mean_field.json"
filenames["ideal"] =  "supplied_data_free_energy_ideal.json"
filenames["hybrid"] =  "supplied_data_free_energy_hybrid.json"


free_energy  = total_free_energy(
    ctx=ctx,
    hc_data=hc_data,
    system_config=system,
    export_json=True,
    filenames = filenames
)


free_energy_exporter(
    ctx = ctx,
    total_fe =  free_energy,
    filename =  "supplied_data_free_energy_total.json",
    system_config=None,
    indent=4,
)




grid =  {}
grid ["r_max"] = 5
grid["n_points"] = 500
grid["r_min"] = 5/500  

densities  = [0.3, 0.001, 0.1]


print (hc_data["sigma"])

value  = coexistence_densities_isochem(
    ctx = ctx,
    config_dict=system,
    fe_res = free_energy,
    supplied_data = None,
    max_outer_iters=10,
    tol_outer=1e-3,
    tol_solver=1e-8,
    verbose=True
)



print (value)

exit (0)

rdf_radial(
    ctx = ctx,
    rdf_config = system,
    grid_dict = grid,
    potential_dict = total_data["total_potentials"],
    densities = densities,
    sigma = hc_data["sigma"],
    supplied_data = None,
    export = False,
    plot = True,
    filename_prefix="rdf",
)





