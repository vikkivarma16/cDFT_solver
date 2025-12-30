import numpy as np
import sympy as sp
import json
from pathlib import Path

def free_energy_EMF_z(ctx=None, hc_data=None, export_json=True, filename="Solution_EMF_z.json"):
    """
    Enhanced Mean-Field (EMF) two-point free-energy kernel:

        f(z, zs) = 1/2 ∑_{i,j} rho_i(z) * v_ij * rho_j(zs) / volume_factor_j(zs)

    Notes
    -----
    • No spatial integration is performed
    • Returned as a symbolic kernel
    • All variables are flattened for Lambda usage
    """

    # -------------------------
    # Validate input
    # -------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

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
    # Volume correction at zs
    # -------------------------
    volume_factor_zs = []
    for j in range(n_species):
        vf = 1
        for i in range(n_species):
            if flag[i] == 1 and i != j:
                avg_p_vol = (0.5 * (sigmai[i] + sigmai[j])) ** 3
                term = (
                    sp.Rational(1, 2) * rho_zs[i] * (sp.pi / 6) * avg_p_vol
                    - sp.Rational(3, 8) * (rho_zs[i] * (sp.pi / 6) * avg_p_vol) ** 2
                )
                vf -= term
        volume_factor_zs.append(vf)

    # -------------------------
    # Two-point EMF kernel
    # -------------------------
    f_emf_z = 0
    for i in range(n_species):
        for j in range(n_species):
            f_emf_z += (
                sp.Rational(1, 2)
                * rho_z[i]
                * vij[i][j]
                * rho_zs[j]
                / volume_factor_zs[j]
            )

    # -------------------------
    # Flatten variables
    # -------------------------
    flat_vars = tuple(
        rho_z +
        rho_zs +
        [vij[i][j] for i in range(n_species) for j in range(n_species)]
    )

    f_emf_z_func = sp.Lambda(flat_vars, f_emf_z)

    # -------------------------
    # Result dictionary
    # -------------------------
    result = {
        "variables": flat_vars,
        "function": f_emf_z_func,
        "expression": f_emf_z,
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
                    "volume_factor": [str(vf) for vf in volume_factor_zs],
                    "variables": [str(v) for v in flat_vars],
                    "function": str(f_emf_z_func),
                    "expression": str(f_emf_z),
                },
                f,
                indent=2,
            )

        print(f"✅ EMF(z,zs) kernel exported: {out_file}")

    return result

