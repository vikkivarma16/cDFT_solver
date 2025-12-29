# cdft_solver/isochores/data_generators/profile_parameters.py

import json
from pathlib import Path

def profile_simulation_configurations(ctx, export_json=True, filename="input_profile_parameters.json"):
    """
    Parse density profile iteration parameters from an input file.

    Parameters
    ----------
    ctx : ExecutionContext
        Must provide ctx.input_file and ctx.scratch_dir
    export_json : bool
        Whether to export the parsed dictionary to JSON
    filename : str
        Output JSON filename (within scratch_dir)
    """

    input_file = Path(ctx.input_file)
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    profile_dict = {}

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Only lines starting with "profile" are considered
            if line.lower().startswith("profile"):
                # Format: profile key = value
                try:
                    _, rest = line.split("profile", 1)
                    key, val = rest.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # convert to int or float if possible
                    try:
                        val_float = float(val)
                        if val_float.is_integer():
                            val = int(val_float)
                        else:
                            val = val_float
                    except:
                        pass
                    profile_dict[key] = val
                except ValueError:
                    print(f"⚠️ Skipping malformed profile line: {line}")
                    continue

    # Export JSON if requested
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump({"profile_parameters": profile_dict}, f, indent=4)
        print(f"✅ Profile parameters exported to: {out_file}")

    return profile_dict


# --- Example standalone usage ---
if __name__ == "__main__":
    from types import SimpleNamespace
    ctx = SimpleNamespace(input_file="interactions.in", scratch_dir=".")
    profile_data = profile_parameters(ctx)
    print(json.dumps(profile_data, indent=2))

