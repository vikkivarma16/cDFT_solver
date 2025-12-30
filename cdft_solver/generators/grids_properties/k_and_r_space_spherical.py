# cdft_solver/isochores/data_generators/r_k_space_spherical_dict.py

"""
Spherical-Symmetry r- and k-space Generator (Dictionary-Driven)

Generates spherical grids (r, theta, phi) from a dictionary containing
confinement parameters. Recursively searches for `space_confinement_parameters`.
Outputs a JSON-ready dictionary and optionally writes to `ctx.scratch_dir`.
"""

import json
import numpy as np
from pathlib import Path


def r_k_space_spherical(
    ctx=None,
    data_dict=None,
    export_json=True,
    filename="supplied_data_r_k_space_spherical.json"
):
    """
    Generate spherical r-space and k-space grids from a dictionary.

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
            "r_space": [[r, theta, phi], ...],
            "k_space": [[kr, ktheta, kphi], ...],
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
    # Spherical r-space
    # -------------------------
    Nr = box_points[0]
    r = np.linspace(0.0, box_length[0], Nr)

    if dimension >= 2:
        Ntheta = box_points[1]
        theta = np.linspace(0.0, np.pi, Ntheta)
    else:
        theta = np.array([0.0])

    if dimension == 3:
        Nphi = box_points[2]
        phi = np.linspace(0.0, 2*np.pi, Nphi, endpoint=False)
    else:
        phi = np.array([0.0])

    R, THETA, PHI = np.meshgrid(r, theta, phi, indexing="ij")
    r_space = np.column_stack([R.ravel(), THETA.ravel(), PHI.ravel()]).tolist()

    # -------------------------
    # Spherical k-space
    # -------------------------
    dr = box_length[0] / (Nr - 1)
    kr = np.fft.fftfreq(Nr, d=dr) * 2.0 * np.pi

    if dimension >= 2:
        dtheta = np.pi / (Ntheta - 1)
        ktheta = np.fft.fftfreq(Ntheta, d=dtheta) * 2.0 * np.pi
    else:
        ktheta = np.array([0.0])

    if dimension == 3:
        dphi = 2.0*np.pi / Nphi
        kphi = np.fft.fftfreq(Nphi, d=dphi) * 2.0 * np.pi
    else:
        kphi = np.array([0.0])

    KR, KTHETA, KPHI = np.meshgrid(kr, ktheta, kphi, indexing="ij")
    k_space = np.column_stack([KR.ravel(), KTHETA.ravel(), KPHI.ravel()]).tolist()

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
        print(f"✅ Spherical r- and k-space grids exported to {out_file}")

    return result

