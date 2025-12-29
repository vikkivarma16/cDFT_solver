# cdft_solver/utils/super_dict.py
import json
from pathlib import Path
import re


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

    Repeated keys → stored as lists
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

    result = base_dict.copy() if base_dict else {}
    result.setdefault(super_key_name, {})

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

    # -------------------------
    # Main parsing loop
    # -------------------------
    with input_file.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            # ====================================
            # PHASE 1 — ATTRIBUTE EXTRACTION
            # ====================================
            segments = [s.strip() for s in line.split(",")]

            attributes = {}
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

            # ====================================
            # PHASE 2 — HIERARCHY EXTRACTION
            # ====================================
            before_eq = line.split("=", 1)[0].strip()

            # remove the first attribute token from the end
            attr_token = attr_order[0] if attr_order else ""
            tokens = before_eq.split()
            if tokens and tokens[-1] == attr_token:
                tokens = tokens[:-1]

            hierarchy_text = " ".join(tokens)

            # split hierarchy by ':' OR whitespace
            hierarchy = [h for h in re.split(r"[:\s]+", hierarchy_text) if h]

            current = result[super_key_name]
            for key in hierarchy[:-1]:
                current = current.setdefault(key, {})

            last_key = hierarchy[-1] if hierarchy else ""

            # -------------------------
            # PROMOTION RULE: attribute-only lines
            # -------------------------
            if not last_key:
                if len(attributes) == 1:
                    promoted_key, promoted_value = next(iter(attributes.items()))
                    current[promoted_key] = promoted_value
                    continue
                else:
                    # fallback to empty string key (rare)
                    last_key = ""

            # handle repeated keys
            if last_key in current:
                if isinstance(current[last_key], list):
                    current[last_key].append(attributes)
                else:
                    current[last_key] = [current[last_key], attributes]
            else:
                current[last_key] = attributes

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out = scratch / filename
        with open(out, "w") as f:
            json.dump(result, f, indent=4)
        print(f"\n✅ Super dictionary exported to: {out}")

    return result


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace

    input_text = """
    species = a, b, c
    interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
    interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
    interaction primary: ac type = ma, sigma = 1.0, cutoff = 3.5, epsilon = 0.1787, m = 12, n = 6, lambda = 0.477246
    interaction secondary: aa type = ghc, sigma = 1.02, cutoff = 3.2, epsilon = 2.0
    profile iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
    """

    tmp = Path("tmp_input.in")
    tmp.write_text(input_text)

    ctx = SimpleNamespace(input_file=tmp, scratch_dir=".")
    d = super_dictionary_creator(ctx, export_json=True)
    print(json.dumps(d, indent=2))

