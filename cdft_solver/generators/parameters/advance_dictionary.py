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
    Universal dictionary builder.

    Correct rules:
    - Split line ONCE at first '='
    - Hierarchy comes ONLY from left part
    - Attributes come ONLY from right part
    - Attributes detected strictly by '='
    - Comma values without '=' belong to previous attribute
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

            # -------------------------
            # 1. Split ONCE at first '='
            # -------------------------
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()

            # -------------------------
            # 2. Extract hierarchy
            # -------------------------
            left_parts = [p.strip() for p in left.split(":")]

            # last hierarchy block may contain attribute word → keep only first token
            last_block_tokens = left_parts[-1].split()
            object_key = last_block_tokens[0]

            hierarchy = left_parts[:-1]

            current = result[super_key_name]
            for h in hierarchy:
                current = current.setdefault(h, {})

            # -------------------------
            # 3. Parse attributes from RIGHT
            # -------------------------
            segments = [s.strip() for s in right.split(",") if s.strip()]
            attrs = {}
            current_attr = None

            for seg in segments:
                if "=" in seg:
                    k, v = seg.split("=", 1)
                    k = k.strip().split()[-1]
                    v = convert_val(v)
                    attrs[k] = [v]
                    current_attr = k
                else:
                    if current_attr is not None:
                        attrs[current_attr].append(convert_val(seg))

            # flatten single-element lists
            for k, v in attrs.items():
                if len(v) == 1:
                    attrs[k] = v[0]

            # -------------------------
            # 4. Assign object
            # -------------------------
            if object_key in current:
                if isinstance(current[object_key], list):
                    current[object_key].append(attrs)
                else:
                    current[object_key] = [current[object_key], attrs]
            else:
                current[object_key] = attrs

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out_file}")

    return result

