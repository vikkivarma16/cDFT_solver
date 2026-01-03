import json
import numpy as np
from pathlib import Path


def load_json_or_dict(obj):
    """
    Accepts either:
    - a dict (returned as-is)
    - a path to a JSON file (loaded and returned)
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, Path)):
        with open(obj, "r") as f:
            return json.load(f)
    raise TypeError("Input must be a dict or a path to a JSON file")


def reconstruct_potential(pot_dict, u_key):
    """
    Convert JSON potential entry to NumPy arrays.
    """
    r = np.asarray(pot_dict["r"], dtype=float)
    u = np.asarray(pot_dict[u_key], dtype=float)
    return r, u
    
    
def find_key_recursive(obj, key):
    """
    Recursively find ALL occurrences of `key` in nested dict/list structures.
    Returns list of matches.
    """
    found = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(find_key_recursive(v, key))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_key_recursive(item, key))

    return found
    
    
    
def build_hc_flag_map(species, flag_matrix):
    """
    Build map: pair -> 0/1
    """
    flag_map = {}
    n = len(species)

    for i in range(n):
        for j in range(n):
            pair = species[i] + species[j]
            flag_map[pair] = int(flag_matrix[i][j])

    return flag_map

def interpolate_to_grid(r_src, u_src, r_target):
    """
    Interpolate u_src(r_src) onto r_target.
    Values outside range → 0.
    """
    return np.interp(
        r_target,
        r_src,
        u_src,
        left=0.0,
        right=0.0
    )



def total_potentials(
    ctx=None,
    hc_source=None,
    mf_source=None,
    file_name_prefix="supplied_data_potential_total.json",
    export_files=True,
    
):
    """
    Assemble hard-core + mean-field potentials with:
      - HC included ONLY if flag == 1
      - grid mismatch handled by interpolation
      - JSON export
    """

    hc_data = load_json_or_dict(hc_source)
    mf_data = load_json_or_dict(mf_source)

    # --------------------------------------------------
    # Robust extraction
    # --------------------------------------------------
    species = find_key_recursive(hc_data, "species")[0]
    hc_pots = hc_data.get("potentials", {})
    mf_pots = mf_data.get("potentials", {})

    flag_matrix = hc_data.get("flag", None)
    if flag_matrix is None:
        raise KeyError("HC data missing 'flag' matrix")

    hc_flag_map = build_hc_flag_map(species, flag_matrix)

    total_pots = {}

    all_pairs = sorted(set(hc_pots) | set(mf_pots))

    for pair in all_pairs:

        r_final = None
        u_total = None

        # ---------------------------
        # Hard-core (ONLY if flag==1)
        # ---------------------------
        if pair in hc_pots and hc_flag_map.get(pair, 0) == 1:
            r_hc, u_hc = reconstruct_potential(hc_pots[pair], "U")
            r_final = r_hc
            u_total = u_hc.copy()
        else:
            r_hc, u_hc = None, None

        # ---------------------------
        # Mean-field
        # ---------------------------
        if pair in mf_pots:
            r_mf, u_mf = reconstruct_potential(mf_pots[pair], "U")
        else:
            r_mf, u_mf = None, None

        # ---------------------------
        # Combine with interpolation
        # ---------------------------
        if r_final is None and r_mf is not None:
            r_final = r_mf
            u_total = u_mf.copy()

        elif r_final is not None and r_mf is not None:
            if not np.allclose(r_final, r_mf):
                u_mf_interp = interpolate_to_grid(r_mf, u_mf, r_final)
            else:
                u_mf_interp = u_mf

            u_total = u_total + u_mf_interp

        if r_final is None:
            continue

        total_pots[pair] = {
            "r": r_final.tolist(),
            "U": u_total.tolist(),
        }

    result = {
        "species": species,
        "hc_flag_map": hc_flag_map,
        "total_potentials": total_pots,
    }

    # --------------------------------------------------
    # Export JSON
    # --------------------------------------------------
    if export_files and ctx is not None:
        out = Path(ctx.scratch_dir) / file_name_prefix
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Exported Total potential to JSON: {out}")

    return result

