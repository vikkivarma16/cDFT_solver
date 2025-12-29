# cdft_solver/utils/super_dict.py

import json
from pathlib import Path
import re
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

    Parameters
    ----------
    ctx : ExecutionContext or None
        If provided, can read ctx.input_file and ctx.scratch_dir
    input_file : str or Path
        If provided, overrides ctx.input_file
    base_dict : dict
        Optional dictionary to update/merge into
    export_json : bool
        Export the resulting dictionary to JSON
    filename : str
        JSON filename (within ctx.scratch_dir)
    super_key_name : str
        Top-level key for the dictionary
    """

    # -----------------------------
    # Determine input file and output dir
    # -----------------------------
    if ctx is not None:
        input_file = input_file or ctx.input_file
        scratch = Path(ctx.scratch_dir)
    else:
        if input_file is None:
            raise ValueError("No input file or ctx provided.")
        scratch = Path(".")
    input_file = Path(input_file)
    scratch.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Initialize dictionary
    # -----------------------------
    result = base_dict.copy() if base_dict else {}
    if super_key_name not in result:
        result[super_key_name] = {}

    # -----------------------------
    # Helper functions
    # -----------------------------
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

            # detect key=value (standard)
            if "=" in line:
                parts = line.split("=")
                # handle multiple "=" in line
                key = parts[0].strip()
                val = "=".join(parts[1:]).strip()

                # detect optional hierarchy via colon
                if ":" in key:
                    key_hierarchy = [k.strip() for k in key.split(":")]
                else:
                    key_hierarchy = [key]

                current = result[super_key_name]

                for i, k in enumerate(key_hierarchy):
                    if i == len(key_hierarchy) - 1:
                        # last key, assign value
                        if k in current:
                            # merge if existing value is dict
                            if isinstance(current[k], dict):
                                # store as special _value key
                                current[k]["_value"] = convert_val(val)
                            else:
                                # overwrite
                                current[k] = convert_val(val)
                        else:
                            current[k] = convert_val(val)
                    else:
                        # intermediate key, ensure dict
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


# --- Example standalone usage ---
if __name__ == "__main__":
    from types import SimpleNamespace

    # minimal example input
    input_text = """
    species = a, b, c
    closure: aa = hnc
    closure: ab = py
    profile iteration_max = 5000
    profile tolerance = 0.00001
    external: d position = 0.0,0.0,0.0
    external: d position = 60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

