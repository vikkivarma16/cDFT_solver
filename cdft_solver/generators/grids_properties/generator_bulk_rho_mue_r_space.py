def bulk_rho_mue_r_space(ctx):
    import numpy as np
    import json
    from pathlib import Path

    # --- Ensure context and paths ---
    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx.scratch_dir must be provided")
    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    json_file_thermo = scratch_dir / "input_data_simulation_thermodynamic_parameters.json"
    r_space_file = scratch_dir / "supplied_data_r_space.txt"
    output_file = scratch_dir / "supplied_data_bulk_mue_rho_r_space.txt"

    # --- Load thermodynamic parameters ---
    with open(json_file_thermo, "r") as f:
        thermo_data = json.load(f)

    # Support nested structure
    if "simulation_thermodynamic_parameters" in thermo_data:
        thermo = thermo_data["simulation_thermodynamic_parameters"]
    else:
        thermo = thermo_data

    species_names = thermo["species"]
    n_species = len(species_names)
    n_phases = thermo["n_phases"]

    rhos_per_phase = np.array(thermo["rhos_per_phase"])
    mues_per_phase = np.array(thermo["mu_per_phase"])
    phase_fractions = np.ones(n_phases) / n_phases  # uniform if not supplied

    # --- Load r-space grid ---
    r_space_data = np.loadtxt(r_space_file)
    if r_space_data.ndim == 1:
        r_space_data = r_space_data.reshape(-1, 3)
    n_points = len(r_space_data)

    # --- Compute phase boundaries along x ---
    x_vals = r_space_data[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    cum_fractions = np.cumsum(phase_fractions)
    cum_fractions[-1] = 1.0
    boundaries = x_min + cum_fractions * (x_max - x_min)

    # --- Assign phase to each point ---
    phase_indices = np.zeros(n_points, dtype=int)
    for i, x in enumerate(x_vals):
        for p, b in enumerate(boundaries):
            if x <= b:
                phase_indices[i] = p
                break

    # --- Build output array ---
    header_cols = ["x", "y", "z"]
    for sp in species_names:
        header_cols.append(f"{sp}_rho")
        header_cols.append(f"{sp}_mue")
    header = " ".join(header_cols)

    output_data = []
    for i, r_point in enumerate(r_space_data):
        p = phase_indices[i]
        row = list(r_point)
        for s_idx in range(n_species):
            row.append(rhos_per_phase[p][s_idx])
            row.append(mues_per_phase[p][s_idx])
        output_data.append(row)

    # --- Save output ---
    np.savetxt(output_file, output_data, fmt="%.18e", header=header)
    print(f"✅ Assigned μ and ρ to r-space grid using {n_phases} phases.")
    print(f"   → Output saved to {output_file}")

    return output_file

