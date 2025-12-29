# cdft_solver/utils/super_dict.py
import json
from pathlib import Path
import re
from collections import OrderedDict



def super_dictionary_creator(
    ctx=None,
    input_file=None,
    base_dict=None,
    export_json=False,
    filename="super_dictionary.json",
    super_key_name="system",
):
    """
    Universal dictionary builder from hierarchical input.

    Algorithm (strictly implemented):

    PHASE 1 — ATTRIBUTE EXTRACTION
    --------------------------------
    • Split line by commas
    • Any segment containing '=' defines a NEW attribute
        - attribute name = last word to the LEFT of '='
        - attribute value = everything to the RIGHT of '='
    • Segments WITHOUT '=' belong to the PREVIOUS attribute
      (multi-value attributes)

    PHASE 2 — HIERARCHY EXTRACTION
    --------------------------------
    • Take text BEFORE the FIRST '='
    • Remove the attribute name (last token)
    • Remaining text defines hierarchy
    • Split hierarchy by ':' or whitespace
    • Last hierarchy key receives attribute dict

    Repeated keys → merged if both dicts; stored as lists only if conflict
    """

    # -------------------------
    # Input handling
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

    result =  {}
    result.setdefault(super_key_name, OrderedDict())

    # -------------------------
    # Helpers
    # -------------------------
    def convert(val):
        val = val.strip()
        try:
            f = float(val)
            return int(f) if f.is_integer() else f
        except:
            return val

    def merge_dicts(existing, new):
        """Merge new dict into existing dict, preserving order."""
        if not isinstance(existing, dict) or not isinstance(new, dict):
            # Conflict, return list
            return [existing, new]

        merged = OrderedDict(existing)
        for k, v in new.items():
            if k in merged:
                merged[k] = merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    # -------------------------
    # Main parsing loop
    # -------------------------
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            # PHASE 1 — ATTRIBUTE EXTRACTION
            segments = [s.strip() for s in line.split(",")]

            attributes = OrderedDict()
            attr_order = []
            current_attr = None

            for seg in segments:
                if "=" in seg:
                    left, right = seg.split("=", 1)
                    attr = left.strip().split()[-1]
                    val = convert(right)

                    attributes[attr] = val
                    attr_order.append(attr)
                    current_attr = attr
                else:
                    # continuation value
                    if current_attr is not None:
                        prev = attributes[current_attr]
                        if not isinstance(prev, list):
                            attributes[current_attr] = [prev]
                        attributes[current_attr].append(convert(seg))

            # PHASE 2 — HIERARCHY EXTRACTION
            before_eq = line.split("=", 1)[0].strip()
            attr_token = attr_order[0] if attr_order else ""
            tokens = before_eq.split()
            if tokens and tokens[-1] == attr_token:
                tokens = tokens[:-1]

            hierarchy_text = " ".join(tokens)
            hierarchy = [h for h in re.split(r"[:\s]+", hierarchy_text) if h]

            current = result[super_key_name]
            for key in hierarchy[:-1]:
                current = current.setdefault(key, OrderedDict())

            last_key = hierarchy[-1] if hierarchy else ""

            # Merge instead of blindly creating a list
            if last_key in current:
                current[last_key] = merge_dicts(current[last_key], attributes)
            else:
                current[last_key] = attributes

    # -------------------------
    # Post-processing: promote attributes to keys if last_key is empty
    # -------------------------
    # Post-processing: promote attributes to keys if last_key is empty
    # -------------------------
    
    def preserve_and_promote(d):
        if not isinstance(d, dict):
            return d

        # Recursively fix child dictionaries first
        for k, v in list(d.items()):
            if isinstance(v, dict):
                d[k] = preserve_and_promote(v)
            elif isinstance(v, list):
                d[k] = [preserve_and_promote(i) if isinstance(i, dict) else i for i in v]

        # Store original order
        original_keys = list(d.keys())
        new_d = OrderedDict()

        for k in original_keys:
            if k == "" and isinstance(d[k], dict):
                v = d[k]
                # Only promote if there is exactly one key inside
                if len(v) == 1:
                    single_key, single_val = next(iter(v.items()))
                    new_d[single_key] = single_val
                else:
                    # Multiple global attributes → attach directly to super_key
                    for attr_key, attr_val in v.items():
                        new_d[attr_key] = attr_val
            else:
                new_d[k] = d[k]

        return new_d

    # Apply to result
    final_result = preserve_and_promote(result[super_key_name])

    result  = {}
    result[super_key_name] =  final_result
    
    
    #result[super_key_name] = preserve_and_promote(result[super_key_name])
    
    
    def update_from_base_recursive(result, base):
        """
        Recursively search `result` for keys in `base`.
        If found, overwrite attributes with `base` values.
        """
        if not isinstance(result, dict):
            return

        for k, v in base.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                # recursively update matching dict
                update_from_base_recursive(result[k], v)
            else:
                # search deeper
                for rk, rv in result.items():
                    if isinstance(rv, dict):
                        update_from_base_recursive(rv, {k: v})
                    elif isinstance(rv, list):
                        for item in rv:
                            if isinstance(item, dict):
                                update_from_base_recursive(item, {k: v})

                pass
       
    if base_dict:
        print ("\nWarning: udpated the dictionary with the supplied values, and overridden the parameters supplied from the input file.")
        update_from_base_recursive(result, base_dict)

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out = scratch / filename
        with open(out, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out}")

    return result

