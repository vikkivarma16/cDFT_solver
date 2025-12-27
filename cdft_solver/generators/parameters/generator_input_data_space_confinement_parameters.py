# New compact & robust space_confinement_parser.py

import json, re
from pathlib import Path
from types import SimpleNamespace

# ------------------------------ Utilities ------------------------------

def _to_float(x):
    try: return float(x)
    except: return x


def parse_list(val):
    return [v.strip() for v in val.split(',') if v.strip()]


def parse_tuple_list(val):
    # "(0,0,0), (60,0,0)" → [[0,0,0],[60,0,0]]
    parts = re.split(r"\)\s*,\s*\(", val.strip())
    clean = []
    for p in parts:
        p = p.strip().lstrip('(').rstrip(')')
        if p:
            clean.append([_to_float(x) for x in p.split(',')])
    return clean


# ------------------------------ Main parser ------------------------------

def data_exporter_space_confinement_parameters(ctx):
    input_file = Path(ctx.input_file)
    scratch = Path(ctx.scratch_dir)

    # Load species interactions JSON (required)
    pj = scratch / 'input_data_particles_interactions_parameters.json'
    if not pj.exists():
        raise FileNotFoundError("missing input_data_particles_interactions_parameters.json")
    species = json.load(pj.open())['particles_interactions_parameters']['species']

    # defaults
    conf = {
        "space_properties": {
            "dimension": 1,
            "confinement": "pbox"
        },
        "box_properties": {
            "box_length": [10.0,5.0,5.0],
            "box_points": [100,100,100]
        },
        "boundary_type": {"aperiodicity_blocker": "NA"},
        "walls": None,
        "wall_interactions": {"primary":{},"secondary":{},"tertiary":{}}
    }

    walls_flag = {}

    # ------------------------------ Parse lines ------------------------------
    for raw in input_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'): continue

        # Handle aperiodicity
        if line.startswith('aperiodicity_blocker'):
            _, v = re.split(r"[:=]", line, 1)
            conf['boundary_type']['aperiodicity_blocker'] = v.strip()
            if v.strip() == 'wall':
                conf['walls'] = {
                    'particles': [],
                    'positions': [],
                    'normals': []
                }
            continue

        # Generic key=value (ignoring wall/wall_interaction here)
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

        # ------------------------------ wall: KEY = VAL ------------------------------
        if line.startswith('wall:') and conf['boundary_type']['aperiodicity_blocker'] == 'wall':
            _, rest = line.split(':',1)
            key, val = [x.strip() for x in rest.split('=',1)]

            if key == 'particles':
                conf['walls']['particles'] = parse_list(val)
                walls_flag = {s+w:[0,0,0] for s in species for w in conf['walls']['particles']}

            elif key == 'position':
                conf['walls']['positions'] = parse_tuple_list(val)

            elif key in ('orientation','normal'):
                conf['walls']['normals'] = parse_tuple_list(val)
            continue

        # ------------------------------ wall_interaction ------------------------------
        if line.startswith('wall_interaction'):
            # format: wall_interaction: ad = custom_3, sigma=1.0, cutoff=3.2, epsilon=0.1
            _, right = line.split(':',1)
            pair, props = [x.strip() for x in right.split('=',1)]
            
            
            chunks = [c.strip() for c in props.split(',') if c.strip()]

            # first chunk is ALWAYS the type definition (no '=')
            temp = {
                "type": chunks[0],
                "sigma": 1.1,
                "cutoff": 3.4,
                "epsilon": 2.0
            }

            # remaining chunks must be key=value
            for chunk in chunks[1:]:
                if "=" not in chunk:
                    continue
                k,v = [x.strip() for x in chunk.split('=',1)]
                temp[k] = _to_float(v)

            

            if pair not in walls_flag: walls_flag[pair] = [0,0,0]
            slot = walls_flag[pair]
            if slot[0]==0:
                conf['wall_interactions']['primary'][pair] = temp; slot[0]=1
            elif slot[1]==0:
                conf['wall_interactions']['secondary'][pair] = temp; slot[1]=1
            else:
                conf['wall_interactions']['tertiary'][pair] = temp; slot[2]=1
            continue

    # ------------------------------ Write output ------------------------------
    out = scratch / 'input_data_space_confinement_parameters.json'

    out_data = {
        "space_confinement_parameters": {
            "space_properties": conf['space_properties'],
            "box_properties": conf['box_properties']
        }
    }

    if conf['boundary_type']['aperiodicity_blocker'] == 'wall' and conf['walls']:
        conf['walls']['interactions'] = conf['wall_interactions']
        out_data['space_confinement_parameters']['walls_properties'] = conf['walls']

    out.write_text(json.dumps(out_data, indent=4))

    return 0


if __name__ == '__main__':
    ctx = SimpleNamespace(input_file='executor_input.in', scratch_dir='.')
    data_exporter_space_confinement_parameters(ctx)

