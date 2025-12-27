# cdft_package/isochores/data_generators/r_k_space.py

"""
Density Functional Minimizer / r- and k-space Generator (context-driven)

Generates real-space (r-space) and reciprocal-space (k-space) points
based on the confinement properties defined in 'input_data_space_confinement_parameters.json'.
Exports data to the folder specified in ctx.scratch_dir.
"""

import json
import numpy as np
from pathlib import Path

def r_k_space_cartesian(ctx):

    
    """
    Generate r-space and k-space Cartesian grids using the scratch folder from ctx.

    Parameters
    ----------
    ctx : object
        Must have attribute 'scratch_dir' indicating output folder
    """
    if ctx is None or not hasattr(ctx, 'scratch_dir'):
        raise ValueError("ctx.scratch_dir must be provided")

    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    input_file = scratch_dir / 'input_data_space_confinement_parameters.json'
    if not input_file.exists():
        raise FileNotFoundError(f"Input JSON not found in scratch folder: {input_file}")

    # --- Helper functions ---
    def read_input(filename: Path):
        with open(filename, 'r') as f:
            return json.load(f)

    def generate_r_space(lengths, points):
        dim = len(lengths)
        axes = [np.linspace(0, lengths[i], int(points[i])) for i in range(dim)]
        if dim == 1:
            return axes[0], np.zeros_like(axes[0]), np.zeros_like(axes[0])
        elif dim == 2:
            x, y = np.meshgrid(axes[0], axes[1], indexing='ij')
            return x, y, np.zeros_like(x)
        elif dim == 3:
            x, y, z = np.meshgrid(*axes, indexing='ij')
            return x, y, z
        else:
            raise ValueError("Only 1D, 2D, 3D are supported.")

    def generate_k_space(lengths, points):
        dim = len(lengths)
        # --- Accurate k-space computation ---
        if dim == 1:
            dk = lengths[0] / (points[0]-1)
            kx = np.fft.fftfreq(points[0], d=dk)
            return kx, np.zeros_like(kx), np.zeros_like(kx)
        elif dim == 2:
            dkx = lengths[0] / (points[0]-1)
            dky = lengths[1] / (points[1]-1)
            kx = np.fft.fftfreq(points[0], d=dkx) * 2*np.pi
            ky = np.fft.fftfreq(points[1], d=dky) * 2*np.pi
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            return kx_grid, ky_grid, np.zeros_like(kx_grid)
        elif dim == 3:
            dkx = lengths[0] / (points[0]-1)
            dky = lengths[1] / (points[1]-1)
            dkz = lengths[2] / (points[2]-1)
            kx = np.fft.fftfreq(points[0], d=dkx) * 2*np.pi
            ky = np.fft.fftfreq(points[1], d=dky) * 2*np.pi
            kz = np.fft.fftfreq(points[2], d=dkz) * 2*np.pi
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            return kx_grid, ky_grid, kz_grid
        else:
            raise ValueError("Only 1D, 2D, 3D are supported.")

    # --- Main ---
    data = read_input(input_file)
    params = data["space_confinement_parameters"]

    box_length = params["box_properties"]["box_length"]
    num_points = [int(params["box_properties"]["box_points"][i]) for i in range (len(params["box_properties"]["box_points"])) ]
    dimension = params["space_properties"]["dimension"]

    # Generate r-space and k-space
    x, y, z = generate_r_space(box_length[:dimension], num_points[:dimension])
    kx, ky, kz = generate_k_space(box_length[:dimension], num_points[:dimension])

    # Flatten grids for saving
    r_data = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    k_data = np.column_stack((kx.flatten(), ky.flatten(), kz.flatten()))

    # Save to scratch folder
    r_file = scratch_dir / "supplied_data_r_space.txt"
    k_file = scratch_dir / "supplied_data_k_space.txt"
    np.savetxt(r_file, r_data)
    np.savetxt(k_file, k_data)

    print(f"\n... r-space saved to {r_file}")
    print(f"... k-space saved to {k_file}\n")
    


    return r_file, k_file

