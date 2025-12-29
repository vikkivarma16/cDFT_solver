# cdft_solver/utils/super_dict.py
import json
from pathlib import Path
from collections import defaultdict

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

    Parsing Logic:
    --------------
    1. Split each line by comma to get segments.
    2. Detect all key=value pairs in segments.
    3. Remaining left-most part (before first '=') → hierarchy key(s), use colons for nesting.
    4. Assign all attributes to the last hierarchy key.
    5. Multiple definitions of the same hierarchy key are stored as a list.
    """

    # Determine input file and output directory
    if ctx is not None:
        input_file = input_file or ctx.input_file
        scratch = Path(ctx.scratch_dir)
    else:
        if input_file is None:
            raise ValueError("No input_file or ctx provided")
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

    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            # Split by comma first
            segments = [seg.strip() for seg in line.split(",")]

            attr_dict = {}
            hierarchy_key_candidate = None

            for seg in segments:
                if "=" in seg:
                    k, v = [s.strip() for s in seg.split("=", 1)]
                    attr_dict[k] = convert_val(v)
                else:
                    # If segment has no '=', it might be hierarchy part
                    if hierarchy_key_candidate is None:
                        hierarchy_key_candidate = seg

            # If hierarchy key candidate contains colon, split for nested dict
            hierarchy_keys = hierarchy_key_candidate.split(":") if hierarchy_key_candidate else ["unnamed"]
            current = result[super_key_name]
            for k in hierarchy_keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            last_key = hierarchy_keys[-1]

            # Handle multiple definitions of the same last_key
            if last_key in current:
                if isinstance(current[last_key], list):
                    current[last_key].append(attr_dict)
                else:
                    current[last_key] = [current[last_key], attr_dict]
            else:
                current[last_key] = attr_dict

    # Export JSON if requested
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result


# --- Example Usage ---
if __name__ == "__main__":
    from types import SimpleNamespace

    input_text = """
    species = a, b, c
    interaction primary: aa: type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
    interaction primary: ab: type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
    interaction secondary: aa: type = ghc, sigma = 1.02, cutoff = 3.2, epsilon = 2.0
    profile: iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
    external: d: position = 0.0,0.0,0.0
    external: d: position = 60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

