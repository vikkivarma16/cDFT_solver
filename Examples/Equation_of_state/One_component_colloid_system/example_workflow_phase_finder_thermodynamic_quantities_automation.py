from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials

from cdft_solver.calculators.total_free_energy.total_free_energy import total_free_energy
from cdft_solver.calculators.total_free_energy.free_energy_exporter import free_energy_exporter
from cdft_solver.calculators.phase_finder.thermodynamic_potentials_isocore import evaluate_canonical_state

import numpy as np
import json
from tqdm import tqdm


# =========================================================
# Setup
# =========================================================
scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)

ctx = ExecutionContext(
    input_file="example_input_phase_finder_isocore.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system = super_dictionary_creator(
    ctx,
    export_json=True,
    filename="input_system.json",
    super_key_name="system"
)


# =========================================================
# Generate potentials
# =========================================================
hc_data = hard_core_potentials(ctx, system, 5000, "hc.json", True)
mf_data = meanfield_potentials(ctx, system, 5000, "mf.json", True)
raw_data = raw_potentials(ctx, system, 5000, "raw.json", True)

total_data = total_potentials(
    ctx,
    hc_source=hc_data,
    mf_source=mf_data,
    file_name_prefix="total.json",
    export_files=True
)


# =========================================================
# Free energy
# =========================================================
filenames = {
    "hard_core": "fe_hc.json",
    "mean_field": "fe_mf.json",
    "ideal": "fe_id.json",
    "hybrid": "fe_hybrid.json",
}

free_energy = total_free_energy(
    ctx=ctx,
    hc_data=hc_data,
    system_config=system,
    export_json=True,
    filenames=filenames
)

free_energy_exporter(
    ctx=ctx,
    total_fe=free_energy,
    filename="fe_total.json",
    indent=4
)


# =========================================================
# Density grid (FROM YOUR DATA)
# second column in your table
# =========================================================
rho_vals = np.array([ 0.047829631 ,
 0.092051503 ,
 0.131009262 ,
 0.162611908, 
 0.196647877, 
 0.225118437, 
 0.326015639 ,
 0.362218,
 0.396196, 
 0.426810,
 0.453910 ,
 0.478527 , 
 0.641778 ,  
 0.734988, 
 0.798874 ,
 0.847508 ,
])

# =========================================================
# EOS loop
# =========================================================
results = []

with tqdm(total=len(rho_vals), desc="Computing EOS") as pbar:

    for rho in rho_vals:

        # Set single-component density
        # assuming species name is "a"
        system["system"]["solution"]["extrinsic_constraints"]["density"]["a"] = float(rho)

        result = evaluate_canonical_state(
            ctx=ctx,
            config_dict=system,
            fe_res=free_energy,
            supplied_data=None,
            export=False,
            verbose=False,
        )

        results.append({
            "rho": float(rho),
            "vij": result["vij"],
            "mu": result["chemical_potentials"][0],
            "pressure": result["pressure"],
        })

        pbar.update(1)


# =========================================================
# Save EOS data
# =========================================================
output_file = "eos_rho_data.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ EOS data saved")
print(f"📁 File: {output_file}")
