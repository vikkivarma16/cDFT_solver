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
    Universal dictionary builder using a strict two-pass algorithm:

    PASS 1 (attributes):
      - Split line by commas
      - Any segment containing '=' defines an attribute
      - Attribute name = last token before '='
      - Values without '=' belong to the previous attribute

    PASS 2 (hierarchy):
      - Everything BEFORE the first '=' contains hierarchy
      - Last token before '=' is NOT hierarchy (it is attribute name)
      - Remaining tokens define hierarchy via ':'
    """

    # -------------------------
    # Resolve input & output
    # -------------------------
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
    result.setdefault(super_key_name, {})

    def convert_val(v):
        v = v.strip()
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except:
            return v

    # -------------------------
    # Parse file
    # -------------------------
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            # ==========================================================
            # PASS 1: Extract attributes FIRST
            # ==========================================================
            segments = [s.strip() for s in line.split(",") if s.strip()]
            attributes = {}
            current_attr = None

            for seg in segments:
                if "=" in seg:
                    left, right = seg.split("=", 1)
                    attr = left.strip().split()[-1]
                    val = convert_val(right)
                    attributes[attr] = [val]
                    current_attr = attr
                else:
                    # continuation value
                    if current_attr is not None:
                        attributes[current_attr].append(convert_val(seg))

            # flatten lists of length 1
            for k, v in attributes.items():
                if len(v) == 1:
                    attributes[k] = v[0]

            # ==========================================================
            # PASS 2: Extract hierarchy
            # ==========================================================
            left_of_first_equal = line.split("=", 1)[0].strip()

            # remove attribute token at end
            tokens = left_of_first_equal.split()
            hierarchy_str = " ".join(tokens[:-1])

            hierarchy = [h.strip() for h in hierarchy_str.split(":") if h.strip()]
            if not hierarchy:
                continue

            current = result[super_key_name]
            for h in hierarchy[:-1]:
                current = current.setdefault(h, {})

            final_key = hierarchy[-1]

            # ==========================================================
            # Assign attributes
            # ==========================================================
            if final_key in current:
                if isinstance(current[final_key], list):
                    current[final_key].append(attributes)
                else:
                    current[final_key] = [current[final_key], attributes]
            else:
                current[final_key] = attributes

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result

