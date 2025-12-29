import json
import numpy as np
from pathlib import Path


def r_k_space_spherical(ctx):
    """
    Generate spherical-symmetry grids (r, theta, phi) based on confinement input.

    Dimension behavior
    ------------------
    dim = 1 : r only (theta = 0, phi = 0)
    dim = 2 : r, theta (phi = 0)
    dim = 3 : r, theta, phi

    Output
    ------
    supplied_data_r_space.txt : columns [r, theta, phi]
    supplied_data_k_space.txt : columns [kr, ktheta, kphi]

    All files written to ctx.scratch_dir
    """

    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx.scratch_dir must be provided")

    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    input_file = scratch / "input_data_space_confinement_parameters.json"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing confinement file: {input_file}")

    # --------------------------------------------------
    # Load confinement parameters
    # --------------------------------------------------
    with open(input_file, "r") as f:
        data = json.load(f)

    params = data["space_confinement_parameters"]
    dim = int(params["space_properties"]["dimension"])

    box_length = params["box_properties"]["box_length"]
    box_points = params["box_properties"]["box_points"]

    # --------------------------------------------------
    # Build spherical grids
    # --------------------------------------------------
    # r direction (always present)
    Rmax = box_length[0]
    Nr = int(box_points[0])
    r = np.linspace(0.0, Rmax, Nr)

    # theta direction
    if dim >= 2:
        Ntheta = int(box_points[1])
        theta = np.linspace(0.0, np.pi, Ntheta)
    else:
        theta = np.array([0.0])

    # phi direction
    if dim == 3:
        Nphi = int(box_points[2])
        phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    else:
        phi = np.array([0.0])

    # --------------------------------------------------
    # Meshgrid (r, theta, phi)
    # --------------------------------------------------
    R, THETA, PHI = np.meshgrid(r, theta, phi, indexing="ij")

    r_data = np.column_stack([
        R.flatten(),
        THETA.flatten(),
        PHI.flatten()
    ])

    # --------------------------------------------------
    # k-space (FFT-consistent)
    # --------------------------------------------------
    dr = Rmax / (Nr - 1)
    kr = np.fft.fftfreq(Nr, d=dr) * 2.0 * np.pi

    if dim >= 2:
        dtheta = np.pi / (Ntheta - 1)
        ktheta = np.fft.fftfreq(Ntheta, d=dtheta) * 2.0 * np.pi
    else:
        ktheta = np.array([0.0])

    if dim == 3:
        dphi = 2.0 * np.pi / Nphi
        kphi = np.fft.fftfreq(Nphi, d=dphi) * 2.0 * np.pi
    else:
        kphi = np.array([0.0])

    KR, KTHETA, KPHI = np.meshgrid(kr, ktheta, kphi, indexing="ij")

    k_data = np.column_stack([
        KR.flatten(),
        KTHETA.flatten(),
        KPHI.flatten()
    ])

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    r_file = scratch / "supplied_data_r_space.txt"
    k_file = scratch / "supplied_data_k_space.txt"

    np.savetxt(r_file, r_data, header="r theta phi")
    np.savetxt(k_file, k_data, header="kr ktheta kphi")

    print(f"✅ Spherical r-space saved to {r_file}")
    print(f"✅ Spherical k-space saved to {k_file}")

    return r_file, k_file
