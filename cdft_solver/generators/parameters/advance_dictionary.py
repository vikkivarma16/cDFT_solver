# cdft_solver/utils/super_dict.py
import json
from pathlib import Path

def super_dictionary_creator(
    ctx=None,
    input_file=None,
    base_dict=None,
    export_json=False,
    filename="super_dictionary.json",
    super_key_name="system"
):
    """
    Universal dictionary builder from hierarchical input.

    Features:
    - Leftmost key = super key (default: 'system')
    - Hierarchy detected via colons in keys
    - Right-hand side of '=' can have multiple key=value pairs separated by commas
    - Handles repeated keys and merges attributes correctly
    """
    # Determine input file and scratch
    if ctx is not None:
        input_file = input_file or ctx.input_file
        scratch = Path(ctx.scratch_dir)
    else:
        if input_file is None:
            raise ValueError("No input file or ctx provided.")
        scratch = Path(".")
    input_file = Path(input_file)
    scratch.mkdir(parents=True, exist_ok=True)

    # Initialize dictionary
    result = base_dict.copy() if base_dict else {}
    if super_key_name not in result:
        result[super_key_name] = {}

    def convert_val(val):
        val = val.strip()
        try:
            v = float(val)
            if v.is_integer():
                return int(v)
            return v
        except:
            return val

    # Parse input file
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()

            # Handle hierarchy via colons
            hierarchy = [k.strip() for k in left.split(":")]
            current = result[super_key_name]
            for k in hierarchy[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            last_key = hierarchy[-1]

            # Ensure last_key is a dict
            if last_key not in current or not isinstance(current[last_key], dict):
                current[last_key] = {}

            # Split right-hand side by comma
            segments = [seg.strip() for seg in right.split(",") if seg.strip()]

            # Process each key=value segment
            for seg in segments:
                if "=" in seg:
                    k, v = [s.strip() for s in seg.split("=", 1)]
                    current[last_key][k] = convert_val(v)
                else:
                    # Segment without '=', assign as None
                    current[last_key][seg] = None

    # Export JSON if requested
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result


# --- Example usage ---
if __name__ == "__main__":
    from types import SimpleNamespace

    input_text = """
    species = a, b, c
    interaction primary: aa: type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
    interaction primary: ab: type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
    interaction secondary: aa = ghc, sigma = 1.02, cutoff = 3.2, epsilon = 2.0
    profile iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
    external: d position = 0.0,0.0,0.0
    external: d position = 60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

