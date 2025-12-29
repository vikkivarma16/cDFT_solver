# cdft_solver/isochores/data_generators/particles_interactions.py

"""
Density Functional Minimizer / Input Data Generator

- Parses input file with system configuration
- Returns species + interactions dictionary
- Optionally exports JSON to scratch folder
"""

import json
from pathlib import Path


def pair_interactions(
    ctx,
    export_json: bool = True,
    json_filename: str = "input_interactions_parameters.json"
):
    """
    Parse an input file and return particle + interaction data.

    Parameters
    ----------
    ctx : ExecutionContext
        - ctx.input_file: Path to input file (can be None)
        - ctx.scratch_dir: Path to scratch folder
    export_json : bool
        Whether to export JSON file (default True)
    json_filename : str
        Name of exported JSON file (default "input_interactions_parameters.json")

    Returns
    -------
    dict
        {
            "species": [...],
            "interactions": {
                "primary": {...},
                "secondary": {...},
                "tertiary": {...}
            }
        }
    """

    if ctx.input_file is None:
        raise ValueError("No input file provided in ctx.input_file")

    input_file = Path(ctx.input_file)
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper
    def is_float(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    # Initialize
    species_list = []
    interactions_data = {"primary": {}, "secondary": {}, "tertiary": {}}
    flag = {}

    # --- Parse input file ---
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Species definition
            if "=" in line and "interaction" not in line and "wall" not in line:
                key, value = line.split("=", 1)
                if key.strip() == "species":
                    names = [s.strip() for s in value.split(",") if s.strip()]
                    species_list.extend(names)
                    # Initialize interactions
                    for i, n1 in enumerate(names):
                        for j, n2 in enumerate(names):
                            if j >= i:
                                pair = n1 + n2
                                flag[pair] = [0, 0, 0]
                                interactions_data["primary"][pair] = {
                                    "type": "gs", "sigma": 1.1, "cutoff": 3.4, "epsilon": 0.0
                                }

            # Interaction lines
            elif "interaction" in line and "wall" not in line:
                _, props_str = line.split(":", 1)
                inter_key, _ = line.split("=", 1)
                _, pair = inter_key.split(":", 1)
                pair = pair.strip()
                props = [p.strip() for p in props_str.split(",") if p.strip()]

                if pair in flag:
                    # Determine tier
                    if flag[pair][0] == 0:
                        tier, idx = "primary", 0
                    elif flag[pair][1] == 0:
                        tier, idx = "secondary", 1
                    elif flag[pair][2] == 0:
                        tier, idx = "tertiary", 2
                    else:
                        raise ValueError(f"Too many interactions for pair {pair}")

                    flag[pair][idx] = 1
                    temp = {"type": "gs", "sigma": 1.1, "cutoff": 3.4, "epsilon": 0.0}

                    for prop in props:
                        if "=" not in prop:
                            continue
                        k, v = [s.strip() for s in prop.split("=", 1)]
                        if k != pair:
                            temp[k] = float(v) if is_float(v) else v
                        else:
                            temp["type"] = v
                        if temp.get("type") == "hc":
                            temp["cutoff"] = temp.get("sigma", 1.0)

                    interactions_data[tier][pair] = temp

    final_data = {"species": species_list, "interactions": interactions_data}

    # Export JSON if requested
    if export_json:
        out_file = output_dir / json_filename
        with open(out_file, "w") as f:
            json.dump({"particles_interactions_parameters": final_data}, f, indent=4)
        print(f"✅ Interaction parameters exported to: {out_file}")

    return final_data


# --- CLI usage ---
if __name__ == "__main__":
    import argparse
    from cdft_solver.utils import create_unique_scratch_dir

    parser = argparse.ArgumentParser(description="Export particles + interactions JSON")
    parser.add_argument("input_file", type=str, help="Path to input file")
    args = parser.parse_args()

    scratch_dir = create_unique_scratch_dir()

    class Ctx:
        input_file = Path(args.input_file)
        scratch_dir = scratch_dir

    ctx = Ctx()
    data_exporter_particles_interactions_parameters(ctx)

