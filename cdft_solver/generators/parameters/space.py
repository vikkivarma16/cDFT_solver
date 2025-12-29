# cdft_solver/isochores/data_generators/space_confinement.py

import json, re
from pathlib import Path

# ------------------------------ Utilities ------------------------------

def _to_float(x):
    try:
        return float(x)
    except:
        return x

def parse_list(val):
    return [v.strip() for v in val.split(',') if v.strip()]

def parse_tuple_list(val):
    parts = re.split(r"\)\s*,\s*\(", val.strip())
    clean = []
    for p in parts:
        p = p.strip().lstrip('(').rstrip(')')
        if p:
            clean.append([_to_float(x) for x in p.split(',')])
    return clean

# ------------------------------ Main parser ------------------------------

def space_confinement(ctx, export_json=True, output_name="input_space_confinement_parameters.json"):
    """
    Parse executor input and build space confinement dictionary.

    Parameters
    ----------
    ctx : ExecutionContext
        Must have ctx.input_file (path to input) and ctx.scratch_dir (output dir)
    export_json : bool
        Whether to export JSON file
    output_name : str
        Output JSON filename

    Returns
    -------
    dict
        Space confinement parameters dictionary
    """

    input_file = Path(ctx.input_file)
    scratch = Path(ctx.scratch_dir)


    # Default configuration
    conf = {
        "space_properties": {"dimension": 1, "confinement": "pbox"},
        "box_properties": {"box_length": [10.0,5.0,5.0], "box_points": [100,100,100]},
        "boundary_type": {"aperiodicity_blocker": "NA"},
    }

    walls_flag = {}

    # Parse lines
    for raw in input_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        # Aperiodicity
        if line.startswith('aperiodicity_blocker'):
            _, v = re.split(r"[:=]", line, 1)
            conf['boundary_type']['aperiodicity_blocker'] = v.strip()
            if v.strip() == 'wall':
                conf['walls'] = {'particles': [], 'positions': [], 'normals': []}
            continue

        # Generic key=value
        if '=' in line and 'wall' not in line:
            key, val = [x.strip() for x in line.split('=',1)]
            if key == 'space_dimension':
                conf['space_properties']['dimension'] = int(val)
            elif key == 'space_confinement':
                conf['space_properties']['confinement'] = val
            elif key == 'box_extension':
                conf['box_properties']['box_length'] = [float(x) for x in val.split(',')]
            elif key == 'box_points':
                conf['box_properties']['box_points'] = [int(float(x)) for x in val.split(',')]
            continue

    # Build final dictionary
    out_dict = {
        "space_confinement_parameters": {
            "space_properties": conf['space_properties'],
            "box_properties": conf['box_properties']
        }
    }

    # Export JSON if requested
    if export_json:
        out_file = Path(scratch) / output_name
        out_file.write_text(json.dumps(out_dict, indent=4))
        print(f"✅ Space confinement exported to: {out_file}")

    return out_dict

# ------------------------------ Standalone test ------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace
    ctx = SimpleNamespace(input_file='executor_input.in', scratch_dir='.')
    space_confinement(ctx)

