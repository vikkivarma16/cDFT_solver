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

    Attribute parsing rules:
    - Attributes are detected strictly by '='
    - Attribute name = last token before '='
    - Values following without '=' belong to the last attribute
    - Multiple values become a list
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

    def convert_val(val):
        val = val.strip()
        try:
            v = float(val)
            return int(v) if v.is_integer() else v
        except:
            return val

    # -------------------------
    # Parse file
    # -------------------------
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            segments = [s.strip() for s in line.split(",")]

            hierarchy_parts = []
            attr_dict = {}
            current_attr = None

            for seg in segments:
                if "=" in seg:
                    # New attribute starts
                    left, right = seg.split("=", 1)
                    attr = left.strip().split()[-1]
                    val = convert_val(right)

                    if isinstance(val, list):
                        attr_dict[attr] = val
                    else:
                        attr_dict[attr] = [val]

                    current_attr = attr
                else:
                    # Continuation of previous attribute
                    if current_attr is not None:
                        attr_dict[current_attr].append(convert_val(seg))
                    else:
                        hierarchy_parts.append(seg)

            # Convert single-value lists to scalars
            for k, v in list(attr_dict.items()):
                if len(v) == 1:
                    attr_dict[k] = v[0]

            # -------------------------
            # Build hierarchy
            # -------------------------
            hierarchy_str = " ".join(hierarchy_parts)
            hierarchy = [h.strip() for h in hierarchy_str.split(":")]

            current = result[super_key_name]
            for key in hierarchy[:-1]:
                current = current.setdefault(key, {})

            last_key = hierarchy[-1]

            # -------------------------
            # Assign attributes
            # -------------------------
            if last_key in current:
                if isinstance(current[last_key], list):
                    current[last_key].append(attr_dict)
                else:
                    current[last_key] = [current[last_key], attr_dict]
            else:
                current[last_key] = attr_dict

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result

