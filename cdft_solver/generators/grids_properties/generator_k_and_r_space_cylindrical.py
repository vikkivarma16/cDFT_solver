import json
import numpy as np
from pathlib import Path


def r_k_space_cylindrical(ctx):
    """
    Generate cylindrical-symmetry grids (z, r, phi) based on confinement input.

    Dimension behavior
    ------------------
    dim = 1 : z only (r = 0, phi = 0)
    dim = 2 : z, r (phi = 0)
    dim = 3 : z, r, phi

    Output
    ------
    supplied_data_r_space.txt : columns [z, r, phi]
    supplied_data_k_space.txt : columns [kz, kr, kphi]

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
    # Build grids
    # --------------------------------------------------
    # z direction (always present)
    Lz = box_length[0]
    Nz = int(box_points[0])
    z = np.linspace(0.0, Lz, Nz)

    # r direction
    if dim >= 2:
        Rmax = box_length[1]
        Nr = int(box_points[1])
        r = np.linspace(0.0, Rmax, Nr)
    else:
        r = np.array([0.0])

    # phi direction
    if dim == 3:
        Nphi = int(box_points[2])
        phi = np.linspace(0.0, 2.0 * np.pi, Nphi, endpoint=False)
    else:
        phi = np.array([0.0])

    # --------------------------------------------------
    # Meshgrid (z, r, phi)
    # --------------------------------------------------
    Z, R, PHI = np.meshgrid(z, r, phi, indexing="ij")

    r_data = np.column_stack([
        Z.flatten(),
        R.flatten(),
        PHI.flatten()
    ])

    # --------------------------------------------------
    # k-space (FFT-consistent)
    # --------------------------------------------------
    dz = Lz / (Nz - 1)
    kz = np.fft.fftfreq(Nz, d=dz) * 2.0 * np.pi

    if dim >= 2:
        dr = Rmax / (Nr - 1)
        kr = np.fft.fftfreq(Nr, d=dr) * 2.0 * np.pi
    else:
        kr = np.array([0.0])

    if dim == 3:
        dphi = 2.0 * np.pi / Nphi
        kphi = np.fft.fftfreq(Nphi, d=dphi) * 2.0 * np.pi
    else:
        kphi = np.array([0.0])

    KZ, KR, KPHI = np.meshgrid(kz, kr, kphi, indexing="ij")

    k_data = np.column_stack([
        KZ.flatten(),
        KR.flatten(),
        KPHI.flatten()
    ])

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    r_file = scratch / "supplied_data_r_space.txt"
    k_file = scratch / "supplied_data_k_space.txt"

    np.savetxt(r_file, r_data, header="z r phi")
    np.savetxt(k_file, k_data, header="kz kr kphi")

    print(f"✅ Cylindrical r-space saved to {r_file}")
    print(f"✅ Cylindrical k-space saved to {k_file}")

    return r_file, k_file

