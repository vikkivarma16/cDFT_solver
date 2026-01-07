# cdft_solver/isochores/data_generators/r_k_space_cylindrical_dict.py

"""
Cylindrical-Symmetry r- and k-space Generator (Dictionary-Driven)

Generates cylindrical grids (z, r, phi) based on confinement parameters
provided in a dictionary. Recursively searches for `space_confinement_parameters`.
Outputs a JSON-ready dictionary and optionally writes to `ctx.scratch_dir`.
"""

import json
import numpy as np
from pathlib import Path


def r_k_space_cylindrical(
    ctx=None,
    data_dict=None,
    export_json=True,
    filename="supplied_data_r_k_space_cylindrical.json"
):
    """
    Generate cylindrical r-space and k-space grids from a dictionary.

    Parameters
    ----------
    ctx : object, optional
        Provides `ctx.scratch_dir` for exporting JSON file.
    data_dict : dict
        Dictionary containing `space_confinement_parameters`.
    export_json : bool
        If True, export the result to a JSON file.
    filename : str
        Name of the JSON file if exported.

    Returns
    -------
    dict
        {
            "r_space": [[z, r, phi], ...],
            "k_space": [[kz, kr, kphi], ...],
            "box_length": [...],
            "box_points": [...],
            "dimension": int
        }
    """

    if data_dict is None:
        raise ValueError("A valid input dictionary must be provided.")

    # -------------------------
    # Recursive key search
    # -------------------------
    def find_key_recursive(d, key):
        if not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        return None

    params = find_key_recursive(data_dict, "space_confinement_parameters")
    if params is None:
        raise KeyError("Could not find 'space_confinement_parameters' in the dictionary.")

    box_length = params["box_properties"]["box_length"]
    box_points = [int(p) for p in params["box_properties"]["box_points"]]
    dimension = int(params["space_properties"]["dimension"])

    # -------------------------
    # Cylindrical r-space
    # -------------------------
    # z-axis
    Nz = box_points[0]
    z = np.linspace(0.0, box_length[0], Nz)

    # r-axis
    if dimension >= 2:
        Nr = box_points[1]
        r = np.linspace(0.0, box_length[1], Nr)
    else:
        r = np.array([0.0])

    # phi-axis
    if dimension == 3:
        Nphi = box_points[2]
        phi = np.linspace(0.0, 2.0*np.pi, Nphi, endpoint=False)
    else:
        phi = np.array([0.0])

    Z, R, PHI = np.meshgrid(z, r, phi, indexing="ij")
    r_space = np.column_stack([Z.ravel(), R.ravel(), PHI.ravel()]).tolist()

    # -------------------------
    # Cylindrical k-space
    # -------------------------
    kz = np.fft.fftfreq(Nz, d=box_length[0]/(Nz-1)) 
    if dimension >= 2:
        kr = np.fft.fftfreq(Nr, d=box_length[1]/(Nr-1))
    else:
        kr = np.array([0.0])
    if dimension == 3:
        kphi = np.fft.fftfreq(Nphi, d=2*np.pi/Nphi) * 2*np.pi
    else:
        kphi = np.array([0.0])

    KZ, KR, KPHI = np.meshgrid(kz, kr, kphi, indexing="ij")
    k_space = np.column_stack([KZ.ravel(), KR.ravel(), KPHI.ravel()]).tolist()

    # -------------------------
    # Prepare output dictionary
    # -------------------------
    result = {
        "r_space": r_space,
        "k_space": k_space,
        "box_length": box_length,
        "box_points": box_points,
        "dimension": dimension
    }

    # -------------------------
    # Export JSON if requested
    # -------------------------
    if export_json and ctx is not None:
        scratch_dir = Path(ctx.scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Cylindrical r- and k-space grids exported to {out_file}")

    return result

