import numpy as np
import sympy as sp
import json
from pathlib import Path
from scipy.interpolate import interp1d
from collections.abc import Mapping

from cdft_solver.generators.potential_splitter.mf import meanfield_potentials


def vij_radial_kernel(
    ctx,
    config,
    kernel,
    supplied_data= None,
    export_json=False,
    filename="vij_integrated.json",
):
    """
    Compute:
        v_ij = ∫ 4π r² K_ij(r) U_ij(r) dr

    Returns
    -------
    dict with keys:
        species
        vij_symbols
        vij_numeric
    """

    # --------------------------------------------------
    # Utility: recursive config lookup
    # --------------------------------------------------
    def find_key_recursive(obj, key):
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
            for v in obj.values():
                out = find_key_recursive(v, key)
                if out is not None:
                    return out
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                out = find_key_recursive(v, key)
                if out is not None:
                    return out
        return None

    # --------------------------------------------------
    # Read config
    # --------------------------------------------------
    species = find_key_recursive(config, "species")
    beta    = find_key_recursive(config, "beta")

    if species is None:
        raise ValueError("Species list not found in config")

    if beta is None:
        raise ValueError("Beta not found in config")

    n_species = len(species)

    # --------------------------------------------------
    # Symbolic matrix
    # --------------------------------------------------
    vij_symbols = [
        [sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
        for i in range(n_species)
    ]

    vij_numeric = {}
    n_grid=5000,

    # --------------------------------------------------
    # Mean-field potentials
    # --------------------------------------------------
    mf_data = meanfield_potentials(
        ctx=ctx,
        input_data=config,
        grid_points=n_grid,
        file_name_prefix="mf.json",
        export_files=True,
    )

    potential_dict = mf_data["potentials"]

    # --------------------------------------------------
    # Build U_dict[(si, sj)] = {r, U}
    # --------------------------------------------------
    U_dict = {}

    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):

            key_ij = si + sj
            key_ji = sj + si

            pdata = potential_dict.get(key_ij) or potential_dict.get(key_ji)
            if pdata is None:
                raise KeyError(f"Missing MF potential for pair {si}-{sj}")

            r = np.asarray(pdata["r"], dtype=float)
            U = beta * np.asarray(pdata["U"], dtype=float)

            U_dict[(si, sj)] = {"r": r, "U": U}
            U_dict[(sj, si)] = {"r": r, "U": U}

    # --------------------------------------------------
    # Integration loop
    # --------------------------------------------------
    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):

            key = (si, sj)
            rkey = (sj, si)

            if key in kernel:
                ker = kernel[key]
                Udat = U_dict[key]
            elif rkey in kernel:
                ker = kernel[rkey]
                Udat = U_dict[rkey]
            else:
                raise KeyError(f"Missing kernel for pair {si}-{sj}")

            rk = np.asarray(ker["r"], dtype=float)
            K  = np.asarray(ker["values"], dtype=float)

            ru = np.asarray(Udat["r"], dtype=float)
            Uv = np.asarray(Udat["U"], dtype=float)

            # Overlapping domain
            r_lo = max(rk.min(), ru.min())
            r_hi = min(rk.max(), ru.max())

            if r_hi <= r_lo:
                raise ValueError(f"No overlapping r-domain for {si}-{sj}")

            r_common = np.linspace(r_lo, r_hi, n_grid)

            Kc = interp1d(
                rk, K,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(r_common)

            Uc = interp1d(
                ru, Uv,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(r_common)

            vij = float(
                np.trapz(4.0 * np.pi * r_common**2 * Kc * Uc, r_common)
            )

            vij_numeric[key]  = vij
            vij_numeric[rkey] = vij

    # --------------------------------------------------
    # Export U(r) matrices
    # --------------------------------------------------
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    for i, si in enumerate(species):
        for j, sj in enumerate(species[i:], start=i):

            Udat = U_dict[(si, sj)]

            fname = scratch / f"U_{si}_{sj}.npz"
            np.savez(
                fname,
                r=Udat["r"],
                U=Udat["U"],
                pair=f"{si}-{sj}",
            )

    # --------------------------------------------------
    # Optional JSON export
    # --------------------------------------------------
    if export_json:
        out = scratch / filename
        with open(out, "w") as f:
            json.dump(
                {
                    "species": species,
                    "vij": {f"{k[0]}_{k[1]}": v for k, v in vij_numeric.items()},
                },
                f,
                indent=4,
            )

    return {
        "species": species,
        "vij_symbols": vij_symbols,
        "vij_numeric": vij_numeric,
    }

