import sympy as sp
import numpy as np
import json
from pathlib import Path

def free_energy_void_z(ctx=None, hc_data=None, export_json=True, filename="Solution_void_z.json"):
    """
    Cavity Mean-Field (CMF) two-point free-energy kernel:

        f(z, zs) = 1/2 ∑_{i,j} rho_i(z) * v_ij * rho_j(zs)
                   / [ volume_factor_i(z) * volume_factor_j(zs) ]

    Notes
    -----
    • Void-volume corrections applied at BOTH points
    • No spatial integration performed
    • Fully symbolic kernel
    """

    # -------------------------
    # Validate input
    # -------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw  = hc_data.get("flag", [])

    n_species = len(species)
    if n_species == 0:
        raise ValueError("No species provided")

    # -------------------------
    # Extract diagonal sigma / flag
    # -------------------------
    def extract_diagonal(data, n, name="array"):
        arr = np.asarray(data)
        if arr.ndim == 1 and arr.size == n:
            return arr.tolist()
        if arr.ndim == 1 and arr.size == n * n:
            return arr.reshape((n, n)).diagonal().tolist()
        if arr.ndim == 2 and arr.shape == (n, n):
            return arr.diagonal().tolist()
        raise ValueError(f"{name} must be length-{n}, {n}x{n}, or flat length-{n*n}")

    sigmai = [float(s) for s in extract_diagonal(sigma_raw, n_species, "sigma")]
    flag   = [int(f)   for f in extract_diagonal(flag_raw,  n_species, "flag")]

    # -------------------------
    # Coordinates
    # -------------------------
    z, zs = sp.symbols("z zs", real=True)

    # -------------------------
    # Two-point densities
    # -------------------------
    rho_z  = [sp.symbols(f"rho_{s}_z")  for s in species]
    rho_zs = [sp.symbols(f"rho_{s}_zs") for s in species]

    # -------------------------
    # Pair interactions
    # -------------------------
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}")
            for j in range(n_species)]
            for i in range(n_species)]

    # -------------------------
    # Void-volume correction factors
    # -------------------------
    volume_factor_z  = []
    volume_factor_zs = []

    for j in range(n_species):
        vf_z  = 1
        vf_zs = 1
        for i in range(n_species):
            if flag[i] == 1 and i != j:
                avg_p_vol = sigmai[i] ** 3
                term_z  = (sp.pi / 6) * avg_p_vol * rho_z[i]
                term_zs = (sp.pi / 6) * avg_p_vol * rho_zs[i]
                vf_z  -= term_z
                vf_zs -= term_zs
        volume_factor_z.append(vf_z)
        volume_factor_zs.append(vf_zs)

    # -------------------------
    # CMF two-point kernel
    # -------------------------
    f_void_z = 0
    for i in range(n_species):
        for j in range(n_species):
            f_void_z += (
                sp.Rational(1, 2)
                * rho_z[i]
                * vij[i][j]
                * rho_zs[j]
                / (volume_factor_z[i] * volume_factor_zs[j])
            )

    # -------------------------
    # Flatten variables
    # -------------------------
    flat_vars = tuple(
        rho_z +
        rho_zs +
        [vij[i][j] for i in range(n_species) for j in range(n_species)]
    )

    f_void_z_func = sp.Lambda(flat_vars, f_void_z)

    # -------------------------
    # Result dictionary
    # -------------------------
    result = {
        "variables": flat_vars,
        "function": f_void_z_func,
        "expression": f_void_z,
    }

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json and ctx is not None and hasattr(ctx, "scratch_dir"):
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out_file = scratch / filename

        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "sigma_eff": sigmai,
                    "flag": flag,
                    "volume_factor_z":  [str(vf) for vf in volume_factor_z],
                    "volume_factor_zs": [str(vf) for vf in volume_factor_zs],
                    "variables": [str(v) for v in flat_vars],
                    "function": str(f_void_z_func),
                    "expression": str(f_void_z),
                },
                f,
                indent=4,
            )

        print(f"✅ Void CMF(z,zs) kernel exported: {out_file}")

    return result

