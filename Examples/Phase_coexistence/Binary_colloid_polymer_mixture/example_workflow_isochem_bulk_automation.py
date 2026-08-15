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


print ("particle sizes can be printed as::::::::::::::::::::\n\n\n\n", hc_data["sigma"], "\n\n\n\n")



import numpy as np
import matplotlib.pyplot as plt
import copy



# ---------------------------------
# Parameters
# ---------------------------------
d_a = 1.0271066034594893
d_b = 0.9528886708198382



phi_a_list = []
phi_b_list = []
binodal_data = []



i = 0
mu_values = np.linspace(7.0, 10.0, 10)
variable_density_bound =  np.linspace(5, 10, 10) 



for mu in mu_values:

    system_copy = copy.deepcopy(system)
    
    system_copy["system"]["solution"]["intrinsic_constraints"]["chemical_potential"]["a"] = mu
    system_copy["system"]["solution"]["extrinsic_constraints"]["total_density_bound"] = variable_density_bound[i]
    
    i += 1
    
    
    
    print ("Now I am standing here in front of everyone !!!!!!!!!!!!!!")

    value = coexistence_densities_isochem(
        ctx=ctx,
        config_dict=system_copy,
        fe_res=free_energy,
        supplied_data=None,
        max_outer_iters=10,
        tol_outer=1e-3,
        tol_solver=1e-8,
        verbose=False
    )

    print("Result:", value)

    if value is None:
        continue

    rhos = value["rhos_per_phase"]

    # Skip trivial single-phase solutions
    if np.allclose(rhos[0], rhos[1], atol=1e-6):
        continue

    # Extract both coexistence phases
    rho_a1, rho_b1 = rhos[0]
    rho_a2, rho_b2 = rhos[1]

    # Convert to packing fractions
    phi_a1 = rho_a1 * (np.pi / 6.0) * d_a**3
    phi_b1 = rho_b1 * (np.pi / 6.0) * d_b**3

    phi_a2 = rho_a2 * (np.pi / 6.0) * d_a**3
    phi_b2 = rho_b2 * (np.pi / 6.0) * d_b**3

    # Store for plotting
    phi_a_list.extend([phi_a1, phi_a2])
    phi_b_list.extend([phi_b1, phi_b2])

    # Store 4-column data
    binodal_data.append([phi_a1, phi_b1, phi_a2, phi_b2, rho_a1, rho_b1, rho_a2, rho_b2])


# ---------------------------------
# Save binodal data (4 columns)
# ---------------------------------
binodal_array = np.array(binodal_data)

np.savetxt(
    "binodal_data.txt",
    binodal_array,
    header="phi_a_phase1  phi_b_phase1  phi_a_phase2  phi_b_phase2",
    fmt="%.10f"
)

print("Binodal data saved to binodal_data.txt")


# ---------------------------------
# High-resolution plot
# ---------------------------------
plt.figure(figsize=(8,8.5))
plt.plot(phi_a_list, phi_b_list, linewidth=2)

plt.xlabel("Packing fraction φ_a", fontsize=14)
plt.ylabel("Packing fraction φ_b", fontsize=14)
plt.title("Binodal Curve (φ_a vs φ_b)", fontsize=16)

plt.grid(True)
plt.tight_layout()

plt.savefig("binodal_plot.png", dpi=800)   # High resolution
plt.close()

print("High-resolution plot saved as binodal_plot.png")


