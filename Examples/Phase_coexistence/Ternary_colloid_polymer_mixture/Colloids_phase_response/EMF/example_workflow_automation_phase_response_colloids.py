from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from tqdm import tqdm
import numpy as np

from cdft_solver.generators.parameters.advance_dictionary import (
    super_dictionary_creator
)

from cdft_solver.generators.potential_splitter.hc import (
    hard_core_potentials
)

from cdft_solver.generators.potential_splitter.mf import (
    meanfield_potentials
)

from cdft_solver.generators.potential_splitter.total import (
    total_potentials
)

from cdft_solver.generators.potential_splitter.raw import (
    raw_potentials
)

from cdft_solver.calculators.total_free_energy.total_free_energy import (
    total_free_energy
)

from cdft_solver.calculators.total_free_energy.free_energy_exporter import (
    free_energy_exporter
)

from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isocore import (
    coexistence_densities_isocore
)

# ============================================================
# Create scratch directory
# ============================================================

scratch = create_unique_scratch_dir()

plots = scratch / "plots"
plots.mkdir(exist_ok=True)

print("\nScratch directory:")
print(scratch)

# ============================================================
# Execution context
# ============================================================

ctx = ExecutionContext(
    input_file="example_input_isocore.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

# ============================================================
# Build system dictionary
# ============================================================

system = super_dictionary_creator(
    ctx,
    export_json=True,
    filename="input_system.json",
    super_key_name="system"
)

# ============================================================
# Generate potentials
# ============================================================

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
    hc_source=hc_data,
    mf_source=mf_data,
    file_name_prefix="supplied_data_potential_total.json",
    export_files=True,
)

# ============================================================
# Free energy
# ============================================================

filenames = {
    "hard_core": "supplied_data_free_energy_hard_core.json",
    "mean_field": "supplied_data_free_energy_mean_field.json",
    "ideal": "supplied_data_free_energy_ideal.json",
    "hybrid": "supplied_data_free_energy_hybrid.json",
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
    filename="supplied_data_free_energy_total.json",
    system_config=None,
    indent=4,
)

# ============================================================
# Density sweep
# ============================================================

# colloid density range
rho_vals = np.linspace(0.03, 0.4, 10)

# output file
output_file = scratch / "coexistence_density_scan.txt"

# ============================================================
# Write header
# ============================================================

with open(output_file, "w") as f:

    header = (
        "# rho_colloid "
        "phase1_rho_a phase1_rho_b phase1_rho_c "
        "phase2_rho_a phase2_rho_b phase2_rho_c "
        "fraction1 fraction2\n"
    )

    f.write(header)

# ============================================================
# Main loop
# ============================================================

print("\nStarting coexistence-density scan...\n")

for rho in tqdm(rho_vals, desc="Scanning densities"):

    # --------------------------------------------------------
    # Change colloid density in system dictionary
    # --------------------------------------------------------

    system["system"]["solution"]["intrinsic_constraints"][
        "species_fraction"
    ]["c"] = float(rho)

    try:

        # ----------------------------------------------------
        # Compute coexistence
        # ----------------------------------------------------

        value = coexistence_densities_isocore(
            ctx=ctx,
            config_dict=system,
            fe_res=free_energy,
            supplied_data=None,
            max_outer_iters=10,
            tol_outer=1e-3,
            tol_solver=1e-8,
            verbose=False
        )

        # ----------------------------------------------------
        # Extract data
        # ----------------------------------------------------

        fractions = value["fractions"]

        rhos = value["rhos_per_phase"]

        # only keep first two phases
        phase1 = rhos[0]
        phase2 = rhos[1]

        frac1 = fractions[0]
        frac2 = 1- fractions[0]

        # ----------------------------------------------------
        # Save to txt
        # ----------------------------------------------------

        with open(output_file, "a") as f:

            line = (
                f"{rho:.8f} "
                f"{phase1[0]:.8f} "
                f"{phase1[1]:.8f} "
                f"{phase1[2]:.8f} "
                f"{phase2[0]:.8f} "
                f"{phase2[1]:.8f} "
                f"{phase2[2]:.8f} "
                f"{frac1:.8f} "
                f"{frac2:.8f}\n"
            )

            f.write(line)

    except Exception as e:

        print(f"\nFailed at rho = {rho:.6f}")
        print(e)

print("\nDone.")
print(f"\nSaved coexistence data to:\n{output_file}")
