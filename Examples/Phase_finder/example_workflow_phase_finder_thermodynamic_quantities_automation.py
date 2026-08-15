from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials
from cdft_solver.calculators.total_free_energy.total_free_energy import total_free_energy
from cdft_solver.calculators.total_free_energy.free_energy_exporter  import free_energy_exporter

from cdft_solver.calculators.phase_finder.thermodynamic_potentials_isocore import evaluate_canonical_state




# define your directory to export the data and the plots



scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)




# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_phase_finder_isocore.in",
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


print ("\n\n\nParticle's sizes are given as:", hc_data["sigma"], "\n\n\n")



result = evaluate_canonical_state(
    ctx=ctx,
    config_dict=system,
    fe_res =  free_energy,
    supplied_data=None,
    export=True,
    output_file="state_from_density.json",
    verbose=True,
)


print (result)



import numpy as np
import json

import numpy as np
import json
from tqdm import tqdm

# ---------------------------------
# Particle diameters
# ---------------------------------
d_a = 1.0
d_b = 1.0

prefactor_a = (np.pi / 6.0) * d_a**3
prefactor_b = (np.pi / 6.0) * d_b**3

# ---------------------------------
# Packing fraction grid
# ---------------------------------
phi_a_vals = np.linspace(1e-4, 0.4, 100)
phi_b_vals = np.linspace(1e-4, 0.5, 100)

results = []

total_steps = len(phi_a_vals) * len(phi_b_vals)

# ---------------------------------
# Loop over φ-grid with progress bar
# ---------------------------------
with tqdm(total=total_steps, desc="Computing thermodynamic surface") as pbar:

    for phi_a in phi_a_vals:
        for phi_b in phi_b_vals:

            # Convert φ → density
            rho_a = phi_a / prefactor_a
            rho_b = phi_b / prefactor_b

            # Update system densities
            system["system"]["solution"]["extrinsic_constraints"]["density"]["a"] = float(rho_a)
            system["system"]["solution"]["extrinsic_constraints"]["density"]["b"] = float(rho_b)

            result = evaluate_canonical_state(
                ctx=ctx,
                config_dict=system,
                fe_res=free_energy,
                supplied_data=None,
                export=False,
                verbose=False,
            )

            results.append({
                "phi_a": float(phi_a),
                "phi_b": float(phi_b),
                "rho_a": float(rho_a),
                "rho_b": float(rho_b),
                "mu_a": result["chemical_potentials"][0],
                "mu_b": result["chemical_potentials"][1],
                "pressure": result["pressure"],
                "free_energy_density": result["free_energy_density"]
            })

            pbar.update(1)

# ---------------------------------
# Save dataset
# ---------------------------------
output_path = "thermo_surface_phi_data.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("Thermodynamic surface data (φ-grid) saved.")
print(f"Saved to: {output_path}")

exit(0)






