# cdft_solver/isochores/data_generators/simulation_profile_parameters.py

import json
from pathlib import Path


def data_exporter_simulation_profile_parameters(ctx):
    """
    Export simulation profile parameters from input file.

    Extracts:
        - alpha (mixing product for Picard's iteration)
        - iteration_max (number of iterations)
        - log_period (log output period)

    Exports:
        1. JSON file with structured data
        2. TXT file with formatted readable output
    """

    input_file = Path(ctx.input_file)
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize parameters ---
    alpha = None
    iteration_max = None
    log_period = None

    # --- Parse input file ---
    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()  # remove comments
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue

            key = key.strip().lower()
            value = value.strip()

            if key == "alpha":
                try:
                    alpha = float(value)
                except ValueError:
                    raise ValueError(f"Invalid value for alpha: {value}")
            elif key == "iteration_max":
                try:
                    iteration_max = int(value)
                except ValueError:
                    raise ValueError(f"Invalid value for iteration_max: {value}")
            elif key == "log_period":
                try:
                    log_period = int(value)
                except ValueError:
                    raise ValueError(f"Invalid value for log_period: {value}")

    # --- Validation ---
    missing_params = [p for p, v in [("alpha", alpha), ("iteration_max", iteration_max), ("log_period", log_period)] if v is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

    # --- Prepare data ---
    result = {
        "simulation_profile_parameters": {
            "alpha": alpha,
            "iteration_max": iteration_max,
            "log_period": log_period,
        }
    }

    # --- Export JSON ---
    output_file_json = output_dir / "input_data_simulation_profile_parameters.json"
    with open(output_file_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Simulation profile JSON exported to: {output_file_json}")



  

    return output_file_json


# --- Entry point if run as standalone script ---
def main():
    import argparse
    from cdft_solver.utils import get_unique_dir

    parser = argparse.ArgumentParser(description="Export simulation profile parameters (alpha, iteration_max, log_period)")
    parser.add_argument("input_file", type=str, help="Path to input file containing profile parameters")
    args = parser.parse_args()

    scratch_dir = get_unique_dir("scratch")

    class Ctx:
        input_file = args.input_file
        scratch_dir = scratch_dir

    ctx = Ctx()

    # --- Run exporter ---
    data_exporter_simulation_profile_parameters(ctx)


if __name__ == "__main__":
    main()

