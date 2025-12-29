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

    Rules:
    - Leftmost key = super key (default: 'system')
    - Key=value pairs after first '=' can be comma-separated
      - Comma separates multiple key=value only if segment contains '='
      - Otherwise, entire segment is part of the value
    """

    # Determine input file and output dir
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

    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Split leftmost key and rest
            if "=" not in line:
                continue  # skip lines without '='
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()

            # Hierarchy: detect colons in left key
            hierarchy = [k.strip() for k in left.split(":")]
            current = result[super_key_name]
            for k in hierarchy[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            # Process right side
            last_key = hierarchy[-1]

            # If multiple comma-separated segments exist
            segments = [seg.strip() for seg in right.split(",")]
            # If any segment contains '=', treat as multiple key=value
            if any("=" in seg for seg in segments):
                if last_key not in current or not isinstance(current[last_key], dict):
                    current[last_key] = {}
                for seg in segments:
                    if "=" not in seg:
                        continue
                    k, v = [s.strip() for s in seg.split("=", 1)]
                    try:
                        v_float = float(v)
                        if v_float.is_integer():
                            v = int(v_float)
                        else:
                            v = v_float
                    except:
                        pass
                    current[last_key][k] = v
            else:
                # Treat entire right as value
                current[last_key] = right

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
    interaction primary: aa type = gs,  sigma = 1.414, cutoff = 3.5, epsilon = 2.01
    interaction primary: ab type = gs,  sigma = 1.414, cutoff = 3.5, epsilon = 2.5
    profile iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
    external: d position = 0.0,0.0,0.0
    external: d position = 60.0,0.0,0.0
    """

    tmp_file = Path("tmp_input.in")
    tmp_file.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")
    super_dict = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(super_dict, indent=2))

