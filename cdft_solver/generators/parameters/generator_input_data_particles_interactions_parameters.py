# cdft_solver/isochores/data_generators/particles_interactions.py

"""
Density Functional Minimizer / Input Data Generator

- Parses input file with system configuration
- Exports particles + interactions in JSON format
- Can be called via ctx in executor
"""

import json
from pathlib import Path


def data_exporter_particles_interactions_parameters(ctx):
    """
    Parse an input file and export particle + interaction data to JSON.

    Parameters
    ----------
    ctx : ExecutionContext
        - ctx.input_file: Path to executor_input.in
        - ctx.scratch_dir: Path where JSON should be saved
    """

    input_file = Path(ctx.input_file)
    output_dir = Path(ctx.scratch_dir)

    def is_convertible_to_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Initialize data structures
    species_list = []
    interactions_data = {"primary": {}, "secondary": {}, "tertiary": {}}
    flag = {}
    
    

    with open(input_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # --- Handle species line ---
        if "=" in line and "interaction" not in line and "wall" not in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "species":
                species_names = [name.strip() for name in value.split(",") if name.strip()]
                species_list.extend(species_names)

                # initialize interactions for each pair
                for i, name1 in enumerate(species_names):
                    for j, name2 in enumerate(species_names):
                        if j >= i:
                            pair = name1 + name2
                            flag[pair] = [0, 0, 0]
                            interactions_data["primary"][pair] = {
                                "type": "gs",
                                "sigma": 1.1,
                                "cutoff": 3.4,
                                "epsilon": 0.0,
                            }

        # --- Handle interactions ---
        elif "interaction" in line and "wall" not in line:
            interaction, _ = line.split("=", 1)
            _, properties_str = line.split(":", 1)
            _, pair = interaction.split(":", 1)
            pair = pair.strip()
            properties_list = properties_str.split(",")

            if pair in flag:
                # decide which tier (primary/secondary/tertiary) to fill
                if flag[pair][0] == 0:
                    tier, idx = "primary", 0
                elif flag[pair][1] == 0:
                    tier, idx = "secondary", 1
                elif flag[pair][2] == 0:
                    tier, idx = "tertiary", 2
                else:
                    raise ValueError(
                        f"Too many interaction definitions for pair {pair} (max 3 allowed)"
                    )

                flag[pair][idx] = 1

                temp = {
                    "type": "gs",
                    "sigma": 1.1,
                    "cutoff": 3.4,
                    "epsilon": 0.0,
                }

                for prop in properties_list:
                    if "=" not in prop:
                        continue
                    param_key, param_value = prop.split("=", 1)
                    param_key = param_key.strip()
                    param_value = param_value.strip()

                    if param_key != pair:
                        if is_convertible_to_float(param_value):
                            temp[param_key] = float(param_value)
                        else:
                            temp[param_key] = param_value
                        if temp.get("type") == "hc":
                            temp["cutoff"] = temp["sigma"]
                    else:
                        temp["type"] = param_value

                interactions_data[tier][pair] = temp

    # --- Prepare final JSON ---
    input_data_particles_interactions = {
        "species": species_list,   # <-- Only species names now
        "interactions": interactions_data,
    }

    output_file = output_dir / "input_data_particles_interactions_parameters.json"
    with open(output_file, "w") as f:
        json.dump({"particles_interactions_parameters": input_data_particles_interactions}, f, indent=4)

    print(f"\n✅ Interaction parameters exported to: {output_file}\n")

    return output_file


# --- Entry point if run as standalone script ---
if __name__ == "__main__":
    import argparse
    from cdft_solver.utils import get_unique_dir

    parser = argparse.ArgumentParser(description="Export particles and interactions JSON")
    parser.add_argument("input_file", type=str, help="Path to input file")
    args = parser.parse_args()

    scratch_dir = get_unique_dir("scratch")

    class Ctx:
        input_file = args.input_file
        scratch_dir = scratch_dir

    ctx = Ctx()
    data_exporter_particles_interactions_parameters(ctx)

