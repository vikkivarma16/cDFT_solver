# cdft_solver/isochores/data_generators/rdf_parameters.py

import json
from pathlib import Path
import itertools

def rdf_configurations(ctx, export_json=True, filename="input_rdf_parameters.json"):
    """
    Parse RDF parameters from an input file and check consistency.

    Only lines starting with 'rdf' are considered.

    Parameters
    ----------
    ctx : ExecutionContext
        Must provide ctx.input_file and ctx.scratch_dir
    export_json : bool
        Whether to export the parsed dictionary to JSON
    filename : str
        Output JSON filename (within scratch_dir)
    """
    input_file = Path(ctx.input_file)
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    rdf_dict = {
        "species": [],
        "closures": {},
        "parameters": {}
    }

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Only process lines starting with 'rdf'
            if not line.lower().startswith("rdf"):
                continue

            content = line[3:].strip()  # remove 'rdf' prefix

            # --- species line ---
            if content.lower().startswith("species"):
                _, val = content.split("=", 1)
                rdf_dict["species"] = [s.strip() for s in val.split(",") if s.strip()]
                continue

            # --- closure lines ---
            if content.lower().startswith("closure"):
                # Format: closure: ab = hnc,
                _, val = content.split(":", 1)
                pair, closure = val.split("=", 1)
                pair = pair.strip()
                closure = closure.strip().rstrip(",")
                rdf_dict["closures"][pair] = closure
                continue

            # --- global parameters ---
            if "=" in content:
                key, val = content.split("=", 1)
                key = key.strip()
                val = val.strip()
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except:
                    pass
                rdf_dict["parameters"][key] = val

    # --- Check consistency: all pairs must have closures ---
    species = rdf_dict["species"]
    all_pairs = set()
    for i, s1 in enumerate(species):
        for j, s2 in enumerate(species):
            if j >= i:
                all_pairs.add(s1 + s2)

    missing_pairs = [pair for pair in all_pairs if pair not in rdf_dict["closures"]]
    if missing_pairs:
        raise ValueError(
            f"Missing closure definition for the following pairs: {missing_pairs}"
        )

    # --- Export JSON if requested ---
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump({"rdf_parameters": rdf_dict}, f, indent=4)
        print(f"✅ RDF parameters exported to: {out_file}")

    return rdf_dict


# --- Example standalone usage ---
if __name__ == "__main__":
    from types import SimpleNamespace
    ctx = SimpleNamespace(input_file="interactions.in", scratch_dir=".")
    rdf_data = rdf_configurations(ctx)
    print(json.dumps(rdf_data, indent=2))

