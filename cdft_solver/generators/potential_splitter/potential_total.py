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


def total_potentials(
    hc_source,
    mf_source
):
    """
    Assemble hard-core + mean-field potentials.

    Parameters
    ----------
    hc_source : dict or path
        Output of hard_core_potentials_from_dict (or its JSON)
    mf_source : dict or path
        Output of meanfield_potentials_from_dict (or its JSON)
    return_numpy : bool
        If True → returns NumPy arrays
        If False → returns JSON-serializable lists

    Returns
    -------
    dict with keys:
        - species
        - hc_potentials
        - mf_potentials
        - total_potentials
    """

    hc_data = load_json_or_dict(hc_source)
    mf_data = load_json_or_dict(mf_source)

    species = hc_data.get("species", [])

    hc_pots = hc_data.get("potentials", {})
    mf_pots = mf_data.get("potentials", {})

    total_potentials = {}

    for pair in sorted(set(hc_pots) | set(mf_pots)):

        # --- Hard-core ---
        if pair in hc_pots:
            r_hc, u_hc = reconstruct_potential(hc_pots[pair], "U")
        else:
            r_hc, u_hc = None, None

        # --- Mean-field ---
        if pair in mf_pots:
            r_mf, u_mf = reconstruct_potential(mf_pots[pair], "U")
        else:
            r_mf, u_mf = None, None

        # --- Grid consistency check ---
        if r_hc is not None and r_mf is not None:
            if not np.allclose(r_hc, r_mf):
                raise ValueError(f"Grid mismatch for pair {pair}")
            r = r_hc
            u_total = u_hc + u_mf
        elif r_hc is not None:
            r = r_hc
            u_total = u_hc
        elif r_mf is not None:
            r = r_mf
            u_total = u_mf
        else:
            continue

        if return_numpy:
            total_potentials[pair] = {
                "r": r,
                "U": u_total
            }
        else:
            total_potentials[pair] = {
                "r": r.tolist(),
                "U": u_total.tolist()
            }

    return {
        "species": species,
        "hc_potentials": hc_pots,
        "mf_potentials": mf_pots,
        "total_potentials": total_potentials
    }

