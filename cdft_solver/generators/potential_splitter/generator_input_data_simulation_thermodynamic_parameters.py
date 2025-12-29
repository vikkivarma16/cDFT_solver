import json
from pathlib import Path


def data_exporter_simulation_thermodynamic_parameters(ctx, result: dict):
    """
    Export coexistence/thermodynamic parameters for isochores or isochem ensembles.
    Handles multiple species and multiple phases.

    Converts reduced-unit data to real units using:
        - rho_real = rho / (length^3)  [1/Å³]
        - P_real = P * 164.66          [atm]
        - mu_real = mu * R*T           [kJ/mol]
    """

    # --- Setup paths ---
    input_file = Path(ctx.input_file)
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Read length and temperature ---
    length = None
    temperature = None
    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            key, value = key.strip().lower(), value.strip()
            if key == "length":
                length = float(value)
            elif key == "temperature":
                temperature = float(value)

    if length is None:
        raise ValueError("Missing 'length' in input file (needed for density conversion).")
    if temperature is None:
        raise ValueError("Missing 'temperature' in input file (needed for μ conversion).")

    # --- Unit conversion constants ---
    R_kJ_per_molK = 8.314462618e-3  # kJ/(mol·K)
    PRESSURE_CONV = 164.66          # reduced → atm
    MUE_CONV = R_kJ_per_molK * temperature  # μ_reduced → kJ/mol

    # --- Save original JSON ---
    output_file_json = output_dir / "input_data_simulation_thermodynamic_parameters.json"
    with open(output_file_json, "w") as f:
        json.dump({"simulation_thermodynamic_parameters": result}, f, indent=4)

    print(f"\n✅ Original thermodynamic JSON exported to: {output_file_json}")

    # --- Extract data safely ---
    ensemble = result.get("ensemble", "unknown")
    species = result.get("species", [])
    n_phases = result.get("n_phases", 1)
    rhos_per_phase = result.get("rhos_per_phase", [])
    mu_per_phase = result.get("mu_per_phase", [])
    pressure_per_phase = result.get("pressure_per_phase", [])

    # --- TXT report ---
    txt_lines = []
    txt_lines.append("# Simulation Thermodynamic Parameters (Real Units)\n")
    txt_lines.append(f"# Ensemble: {ensemble}")
    txt_lines.append(f"# Number of species: {len(species)}")
    txt_lines.append(f"# Number of phases: {n_phases}")
    txt_lines.append(f"# Using length = {length} Å")
    txt_lines.append(f"# Using temperature = {temperature} K\n")
    txt_lines.append("# Units:")
    txt_lines.append("#   Density: 1/Å³")
    txt_lines.append("#   Pressure: atm")
    txt_lines.append("#   Chemical potential: kJ/mol\n")

    # --- Phase-by-phase loop ---
    for phase_index in range(n_phases):
        txt_lines.append(f"\n=== Phase {phase_index + 1} ===")
        if phase_index < len(rhos_per_phase):
            rho_vals = rhos_per_phase[phase_index]
        else:
            rho_vals = []
        if phase_index < len(mu_per_phase):
            mu_vals = mu_per_phase[phase_index]
        else:
            mu_vals = []
        pressure_val = pressure_per_phase[phase_index] if phase_index < len(pressure_per_phase) else None

        # --- Species data ---
        txt_lines.append("Species:")
        for i, sp in enumerate(species):
            try:
                rho_red = rho_vals[i]
                mu_red = mu_vals[i]
                rho_real = rho_red / (length ** 3)
                mu_real = mu_red * MUE_CONV
                txt_lines.append(f"  {sp}:")
                txt_lines.append(f"    ρ_reduced = {rho_red:.6e}")
                txt_lines.append(f"    ρ_real    = {rho_real:.6e} [1/Å³]")
                txt_lines.append(f"    μ_reduced = {mu_red:.6e}")
                txt_lines.append(f"    μ_real    = {mu_real:.6e} [kJ/mol]")
            except Exception as e:
                txt_lines.append(f"  ⚠️ Error for species {sp}: {e}")

        # --- Pressure data ---
        if pressure_val is not None:
            try:
                p_real = pressure_val * PRESSURE_CONV
                txt_lines.append(f"\nPressure:")
                txt_lines.append(f"  P_reduced = {pressure_val:.6e}")
                txt_lines.append(f"  P_real    = {p_real:.6e} [atm]")
            except Exception as e:
                txt_lines.append(f"  ⚠️ Error converting pressure: {e}")

    # --- Optional metadata ---
    txt_lines.append("\n=== Additional Information ===")
    for key in ["vk_mode_initial", "vk_mode_iter"]:
        if key in result:
            txt_lines.append(f"{key}: {result[key]}")

    # --- Write TXT output ---
    output_file_txt = output_dir / "data_coexistence_thermodynamic_parameters_real_units.txt"
    with open(output_file_txt, "w") as f:
        f.write("\n".join(txt_lines))

    print(f"✅ Real-unit thermodynamic data exported to: {output_file_txt}\n")

    return output_file_json, output_file_txt

