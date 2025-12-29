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

    Handles:
    - Hierarchy with colon
    - Inline attributes after space in last key
      e.g. 'aa type = gs, sigma = 1.414' → {"aa": {"type": "gs", "sigma": 1.414}}
    - Comma-separated key=value pairs
    - Comma-separated values without '=' → list
    - Optional update from base_dict
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

            if "=" not in line:
                continue

            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()

            # Handle hierarchy via colon
            hierarchy = [k.strip() for k in left.split(":")]
            current = result[super_key_name]
            for k in hierarchy[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            last_key = hierarchy[-1]

            # Inline attributes after space in last key
            if " " in last_key:
                tokens = last_key.split()
                last_key = tokens[0]
                inline_attrs = tokens[1:]
                if last_key not in current or not isinstance(current[last_key], dict):
                    current[last_key] = {}
                for attr in inline_attrs:
                    if attr not in current[last_key]:
                        current[last_key][attr] = None
            else:
                inline_attrs = []
                if last_key not in current:
                    current[last_key] = {}

            # Split right-hand side by comma
            segments = [seg.strip() for seg in right.split(",") if seg.strip()]

            # If all segments contain '=', treat as key=value pairs
            if all("=" in seg for seg in segments):
                for seg in segments:
                    k, v = [s.strip() for s in seg.split("=", 1)]
                    v = convert_val(v)
                    current[last_key][k] = v
            else:
                # Some segments do not have '=', treat as list
                values = [convert_val(seg) for seg in segments]
                if inline_attrs:
                    # assign first value to first inline_attr, etc.
                    for val, attr in zip(values, inline_attrs):
                        current[last_key][attr] = val
                else:
                    # if last_key is empty dict, convert to list
                    if isinstance(current[last_key], dict) and not current[last_key]:
                        current[last_key] = values
                    else:
                        # if some inline key=value exists, append as extra values
                        current[last_key]["_values"] = values

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
    interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
    interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
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

