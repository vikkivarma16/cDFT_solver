import numpy as np
import sympy as sp
import json
from pathlib import Path
from scipy.interpolate import interp1d


def vij_radial_kernel(
    ctx=None,
    species,
    kernel,
    U,
    n_grid=2000,
    r_min=None,
    r_max=None,
    export_json=False,
    filename="vij_integrated.json",
   
):
    """
    Compute v_ij = ∫ 4π r^2 K_ij(r) U_ij(r) dr
    with grid mismatch handled by interpolation.

    Parameters
    ----------
    kernel : dict
        {(s1, s2): {"r": array, "values": array}}
    U : dict
        {(s1, s2): {"r": array, "values": array}}
    species : list
        Ordered list of species
    r_min, r_max : float, optional
        Integration bounds (auto-detected if None)
    n_grid : int
        Number of points in common grid
    export_json : bool
        Export numeric results
    filename : str
        JSON output file
    ctx : object, optional
        Must have ctx.scratch_dir if export_json=True

    Returns
    -------
    dict
        {
            "species": [...],
            "vij_symbols": [[sympy.Symbol]],
            "vij_numeric": {(s1, s2): float}
        }
    """

    n_species = len(species)

    # -------------------------
    # Symbolic v_ij matrix
    # -------------------------
    vij_symbols = [
        [sp.symbols(f"v_{species[i]}_{species[j]}")
         for j in range(n_species)]
        for i in range(n_species)
    ]

    vij_numeric = {}

    # -------------------------
    # Loop over species pairs
    # -------------------------
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            key = (si, sj)

            if key not in kernel or key not in U:
                raise KeyError(f"Missing kernel or U for pair {key}")

            rk = np.asarray(kernel[key]["r"], dtype=float)
            K  = np.asarray(kernel[key]["values"], dtype=float)

            ru = np.asarray(U[key]["r"], dtype=float)
            Uv = np.asarray(U[key]["values"], dtype=float)

            # -------------------------
            # Determine common grid
            # -------------------------
            r_lo = max(rk.min(), ru.min())
            r_hi = min(rk.max(), ru.max())

            if r_min is not None:
                r_lo = max(r_lo, r_min)
            if r_max is not None:
                r_hi = min(r_hi, r_max)

            if r_hi <= r_lo:
                raise ValueError(f"No overlapping r-domain for pair {key}")

            r_common = np.linspace(r_lo, r_hi, n_grid)

            # -------------------------
            # Interpolate
            # -------------------------
            K_interp = interp1d(
                rk, K,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )

            U_interp = interp1d(
                ru, Uv,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )

            Kc = K_interp(r_common)
            Uc = U_interp(r_common)

            # -------------------------
            # Radial integral
            # -------------------------
            integrand = 4.0 * np.pi * r_common**2 * Kc * Uc
            vij_val = np.trapz(integrand, r_common)

            vij_numeric[key] = float(vij_val)

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json:
        if ctx is None or not hasattr(ctx, "scratch_dir"):
            raise ValueError("ctx with scratch_dir required for JSON export")

        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out_file = scratch / filename

        with open(out_file, "w") as f:
            json.dump(
                {
                    "species": species,
                    "vij": {f"{k[0]}_{k[1]}": v for k, v in vij_numeric.items()}
                },
                f,
                indent=4,
            )

        print(f"✅ Integrated v_ij exported to {out_file}")

    return {
        "species": species,
        "vij_symbols": vij_symbols,
        "vij_numeric": vij_numeric,
    }

