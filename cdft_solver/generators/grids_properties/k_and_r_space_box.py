# cdft_solver/isochores/data_generators/r_k_space_dict.py

"""
Real- and Reciprocal-Space Grid Generator (Dictionary-Driven)

Generates Cartesian r-space and k-space grids based on confinement
parameters provided in a dictionary. Recursively searches for
`space_confinement_parameters`. Outputs JSON-ready dictionary and
optionally writes to `ctx.scratch_dir`.
"""

import json
import numpy as np
from pathlib import Path


def r_k_space_box(
    ctx=None,
    data_dict=None,
    export_json=True,
    filename="supplied_data_r_k_space_box.json"
):
    """
    Generate r-space and k-space grids from a dictionary.

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
            "r_space": [[x, y, z], ...],
            "k_space": [[kx, ky, kz], ...],
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
    # Grid generators
    # -------------------------
    def generate_r_space(lengths, points):
        dim = len(lengths)
        axes = [np.linspace(0.0, lengths[i], points[i]) for i in range(dim)]

        if dim == 1:
            x = axes[0]
            return x, np.zeros_like(x), np.zeros_like(x)
        elif dim == 2:
            x, y = np.meshgrid(axes[0], axes[1], indexing="ij")
            return x, y, np.zeros_like(x)
        elif dim == 3:
            x, y, z = np.meshgrid(*axes, indexing="ij")
            return x, y, z
        else:
            raise ValueError("Only 1D, 2D, 3D supported.")

    def generate_k_space(lengths, points):
        dim = len(lengths)
        if dim == 1:
            dr = lengths[0] / (points[0]-1)
            kx = np.fft.fftfreq(points[0], d=dr) * 2*np.pi
            return kx, np.zeros_like(kx), np.zeros_like(kx)
        elif dim == 2:
            drx = lengths[0] / (points[0]-1)
            dry = lengths[1] / (points[1]-1)
            kx = np.fft.fftfreq(points[0], d=drx) * 2*np.pi
            ky = np.fft.fftfreq(points[1], d=dry) * 2*np.pi
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
            return kx_grid, ky_grid, np.zeros_like(kx_grid)
        elif dim == 3:
            drx = lengths[0] / (points[0]-1)
            dry = lengths[1] / (points[1]-1)
            drz = lengths[2] / (points[2]-1)
            kx = np.fft.fftfreq(points[0], d=drx) * 2*np.pi
            ky = np.fft.fftfreq(points[1], d=dry) * 2*np.pi
            kz = np.fft.fftfreq(points[2], d=drz) * 2*np.pi
            return np.meshgrid(kx, ky, kz, indexing="ij")
        else:
            raise ValueError("Only 1D, 2D, 3D supported.")

    # -------------------------
    # Compute grids
    # -------------------------
    x, y, z = generate_r_space(box_length[:dimension], box_points[:dimension])
    kx, ky, kz = generate_k_space(box_length[:dimension], box_points[:dimension])

    r_space = np.column_stack((x.ravel(), y.ravel(), z.ravel())).tolist()
    k_space = np.column_stack((kx.ravel(), ky.ravel(), kz.ravel())).tolist()

    # -------------------------
    # Prepare output dictionary
    # -------------------------
    result = {
        "box_length": box_length,
        "box_points": box_points,
        "dimension": dimension,
        "r_space": r_space,
        "k_space": k_space,
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
        print(f"✅ r- and k-space grids exported to {out_file}")

    return result

