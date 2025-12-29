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
    Generic hierarchical dictionary builder from input lines with '=' and optional comma-separated key=value pairs.

    - Works for any input format with key=value and optional colons/spaces hierarchy.
    - Handles multiple key=value pairs separated by comma on the same line.
    - Can merge into an existing base dictionary.
    """
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

    # -----------------------------
    # Parse file
    # -----------------------------
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Handle multiple comma-separated key=value pairs
            # Step 1: Find first '=' in line (for hierarchy detection)
            if "=" not in line:
                continue
            lhs, rhs_all = line.split("=", 1)
            lhs_keys = [k.strip() for k in lhs.replace(":", " ").split() if k.strip()]

            # Step 2: Split rhs into multiple key=value by commas, if any
            rhs_pairs = [rhs_all.strip()]
            if ',' in rhs_all:
                parts = rhs_all.split(',')
                rhs_pairs = [p.strip() for p in parts if p.strip()]

            # Step 3: Assign to dictionary
            current = result[super_key_name]
            for i, k in enumerate(lhs_keys):
                if i == len(lhs_keys) - 1:
                    # last key → assign dict for multiple rhs pairs
                    if len(rhs_pairs) == 1:
                        current[k] = convert_val(rhs_pairs[0])
                    else:
                        # store each comma-separated key=value pair
                        temp_dict = {}
                        for pair in rhs_pairs:
                            if "=" in pair:
                                subk, subv = pair.split("=", 1)
                                temp_dict[subk.strip()] = convert_val(subv)
                            else:
                                # if no '=', store as value list
                                temp_dict[pair] = None
                        current[k] = temp_dict
                else:
                    # intermediate key → ensure dict
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]

    # -----------------------------
    # Export JSON if requested
    # -----------------------------
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result


# --- Example ---
if __name__ == "__main__":
    from types import SimpleNamespace

    input_text = """
    species = a, b, c
    interaction primary: aa type=gs, sigma=1.414, cutoff=3.5, epsilon=2.01
    interaction primary: ab type=gs, sigma=1.414, cutoff=3.5, epsilon=2.5
    profile iteration_max=5000, tolerance=0.00001, alpha=0.1
    external d position=0.0,0.0,0.0
    external d position=60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

