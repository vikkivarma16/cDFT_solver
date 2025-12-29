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
    Generic hierarchical dictionary builder from input lines with '='.

    - Works for any input format with key=value and optional colons/hierarchy.
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

    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Process line by taking the **last '='**
            if "=" not in line:
                continue
            left, right = line.rsplit("=", 1)
            right = convert_val(right.strip())

            # Left-hand side hierarchy
            keys = [k.strip() for k in left.replace(":", " ").split() if k.strip()]
            if not keys:
                continue

            # Assign to dictionary recursively
            current = result[super_key_name]
            for i, k in enumerate(keys):
                if i == len(keys) - 1:
                    # last key → assign value
                    if k in current and isinstance(current[k], dict):
                        # store previous value under "_value" if needed
                        current[k]["_value"] = right
                    else:
                        current[k] = right
                else:
                    # intermediate key → ensure dict
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]

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
    interaction primary: aa type = gs
    interaction primary: ab type = gs
    profile iteration_max = 5000
    profile tolerance = 0.00001
    external d position = 0.0,0.0,0.0
    external d position = 60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

