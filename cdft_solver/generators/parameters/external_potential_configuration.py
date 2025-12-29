# cdft_solver/isochores/data_generators/external_potentials.py

import json
from pathlib import Path

def external_potentials(ctx, export_json=True, filename="input_external_potentials_parameters.json"):
    """
    Parse external potential configuration from input file.

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

    data_dict = {
        "species": [],
        "external_species": None,
        "interactions": {},
        "positions": {}
    }

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # --- species line ---
            if line.lower().startswith("species"):
                _, val = line.split("=", 1)
                data_dict["species"] = [s.strip() for s in val.split(",") if s.strip()]
                continue

            # --- external species ---
            if line.lower().startswith("external species"):
                _, val = line.split("=", 1)
                data_dict["external_species"] = val.strip()
                continue

            # --- external interaction ---
            if line.lower().startswith("external interaction"):
                _, right = line.split(":", 1)
                pair, props = [x.strip() for x in right.split("=", 1)]

                chunks = [c.strip() for c in props.split(",") if c.strip()]
                temp = {
                    "type": chunks[0],  # first chunk is the type
                    "sigma": 1.1,
                    "cutoff": 3.4,
                    "epsilon": 0.0
                }
                # remaining key=value chunks
                for c in chunks[1:]:
                    if "=" not in c:
                        continue
                    k, v = [x.strip() for x in c.split("=", 1)]
                    try:
                        v = float(v)
                    except:
                        pass
                    temp[k] = v

                data_dict["interactions"][pair] = temp
                continue

            # --- external particle positions ---
            if line.lower().startswith("external "):
                parts = line.split()
                if len(parts) < 5:
                    continue
                species_name = parts[1]
                coords = [float(x.strip()) for x in " ".join(parts[2:]).split(",")]
                if species_name not in data_dict["positions"]:
                    data_dict["positions"][species_name] = []
                data_dict["positions"][species_name].append(coords)
                continue

    # --- consistency checks ---
    ext_sp = data_dict["external_species"]
    if not ext_sp:
        raise ValueError("External species not defined in input.")

    if ext_sp not in data_dict["positions"] or len(data_dict["positions"][ext_sp]) == 0:
        raise ValueError(f"No positions provided for external species '{ext_sp}'.")

    # --- export JSON ---
    if export_json:
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump({"external_potentials": data_dict}, f, indent=4)
        print(f"\n✅ External potentials exported to: {out_file}")

    return data_dict


# --- Example standalone usage ---
if __name__ == "__main__":
    from types import SimpleNamespace
    ctx = SimpleNamespace(input_file="interactions.in", scratch_dir=".")
    ext_data = external_potentials_config(ctx)
    import pprint
    pprint.pprint(ext_data)

