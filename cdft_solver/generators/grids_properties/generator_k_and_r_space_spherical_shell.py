# density functional minimizer/r and k space points generator in spherical shell

import json
import numpy as np
import os

def r_k_space_spherical_shell(ctx=None):
    """
    Generate r-space and k-space points in a spherical shell.
    Exports the data to the scratch folder given by ctx.scratch_folder.
    """

    if ctx is None or not hasattr(ctx, 'scratch_folder'):
        raise ValueError("Scratch folder path must be provided via ctx.scratch_folder")

    output_dir = ctx.scratch_folder
    os.makedirs(output_dir, exist_ok=True)

    def read_input(filename):
        """Reads input file and returns parameters as a dictionary."""
        with open(filename, 'r') as f:
            params = json.load(f)
        return params

    def generate_r_space_spherical(radius, n_r, n_theta, n_phi):
        """Generate r, theta, phi grid in spherical coordinates and convert to Cartesian coordinates."""
        r = np.linspace(0, radius, n_r)
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)

        r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')

        x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z = r_grid * np.cos(theta_grid)

        return x, y, z

    def generate_k_space_spherical(radius, n_r, n_theta, n_phi):
        """Generate k-space for spherical grid."""
        dr = radius / (n_r - 1)
        k_r = np.fft.fftfreq(n_r, d=dr) * 2 * np.pi

        dtheta = np.pi / (n_theta - 1)
        dphi = 2*np.pi / n_phi
        k_theta = np.fft.fftfreq(n_theta, d=dtheta) * 2 * np.pi
        k_phi = np.fft.fftfreq(n_phi, d=dphi) * 2 * np.pi

        k_r_grid, k_theta_grid, k_phi_grid = np.meshgrid(k_r, k_theta, k_phi, indexing='ij')

        kx = k_r_grid * np.sin(k_theta_grid) * np.cos(k_phi_grid)
        ky = k_r_grid * np.sin(k_theta_grid) * np.sin(k_phi_grid)
        kz = k_r_grid * np.cos(k_theta_grid)

        return kx, ky, kz

    # Read input parameters (use ctx.input_file if available)
    input_file = getattr(ctx, 'input_file', 'input_data_space_confinement_parameters.json')
    params = read_input(input_file)
    space_params = params['space_confinement_parameters']
    radius = space_params['box_properties']['box_length'][0]
    n_r = int(space_params['box_properties']['box_points'][0])
    n_theta = int(space_params['box_properties']['box_points'][1])
    n_phi = int(space_params['box_properties']['box_points'][2])

    # Generate r-space
    x, y, z = generate_r_space_spherical(radius, n_r, n_theta, n_phi)
    r_file = os.path.join(output_dir, 'r_space_spherical.txt')
    np.savetxt(r_file, np.column_stack((x.flatten(), y.flatten(), z.flatten())))
    print(f"\n... r-space spherical shell data saved to '{r_file}' ...\n")

    # Generate k-space
    kx, ky, kz = generate_k_space_spherical(radius, n_r, n_theta, n_phi)
    k_file = os.path.join(output_dir, 'k_space_spherical.txt')
    np.savetxt(k_file, np.column_stack((kx.flatten(), ky.flatten(), kz.flatten())))
    print(f"\n... k-space spherical shell data saved to '{k_file}' ...\n")

    return 0

# Example usage with ctx:
# ctx.scratch_folder = '/path/to/scratch'
# ctx.input_file = '/path/to/input.json'
# r_k_space_spherical_shell(ctx)

