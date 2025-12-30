import numpy as np
import sympy as sp
import json
from pathlib import Path

def free_energy_void(ctx=None, hc_data=None, export_json=True, filename="Solution_void.json"):
    """
    Computes the symbolic cavity mean-field (CMF) free energy for a multi-species system
    using void corrections.

    Parameters
    ----------
    ctx : object, optional
        Must have `scratch_dir` for exporting JSON.
    hc_data : dict
        Hard-core / species data, e.g.,
        {
            "species": [...],
            "sigma": [...],
            "flag": [...]
        }
    export_json : bool
        Whether to save symbolic results to JSON.
    filename : str
        JSON output filename.

    Returns
    -------
    dict
        {
            "species": [...],
            "densities": [...],
            "vij": [[...], [...]],
            "volume_factors": [...],
            "variables": tuple,
            "f_void_func": sympy.Lambda,
            "f_void": sympy.Expr
        }
    """

    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

    n_species = len(species)
    if n_species == 0:
        raise ValueError("No species provided in hc_data")

    if not (len(species) == len(sigma_raw) == len(flag_raw)):
        raise ValueError("Length mismatch between species, sigma, and flag")

    # -------------------------
    # Extract sigma and flag
    # -------------------------
    def extract_diagonal(data, n, name="array"):
        arr = np.asarray(data)
        if arr.ndim == 1 and arr.size == n:
            return arr.tolist()
        if arr.ndim == 1 and arr.size == n*n:
            return arr.reshape((n,n)).diagonal().tolist()
        if arr.ndim == 2 and arr.shape == (n,n):
            return arr.diagonal().tolist()
        raise ValueError(f"{name} must be length-{n}, {n}x{n}, or flat length-{n*n} array")

    sigmai = [float(s) for s in extract_diagonal(sigma_raw, n_species, name="sigma")]
    flag = [int(f) for f in extract_diagonal(flag_raw, n_species, name="flag")]

    # -------------------------
    # Symbolic densities
    # -------------------------
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # -------------------------
    # Symbolic interactions
    # -------------------------
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
           for i in range(n_species)]

    # -------------------------
    # Compute volume correction factors
    # -------------------------
    volume_factors = []
    for j in range(n_species):
        factor = 0
        for i in range(n_species):
            if flag[i] == 1 and i != j:
                avg_p_vol = sigmai[i] ** 3
                term = (sp.pi / 6) * avg_p_vol * densities[i]
                factor += term
        volume_factors.append(1 - factor)

    # -------------------------
    # Symbolic CMF free energy
    # -------------------------
    f_void = 0
    for i in range(n_species):
        for j in range(n_species):
            f_void += sp.Rational(1, 2) * vij[i][j] * densities[i] * densities[j] / (
                volume_factors[i] * volume_factors[j]
            )

    # -------------------------
    # Flatten all variables for Lambda
    # -------------------------
    flat_vars = tuple(densities + [vij[i][j] for i in range(n_species) for j in range(n_species)])
    f_void_func = sp.Lambda(flat_vars, f_void)

    # -------------------------
    # Prepare result
    # -------------------------
    result = {
        "variables": flat_vars,
        "function": f_void_func,
        "expression": f_void,
    }

    # -------------------------
    # Optional JSON export
    # -------------------------
    if export_json and ctx is not None and hasattr(ctx, "scratch_dir"):
        scratch = Path(ctx.scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        out_file = scratch / filename
        with open(out_file, "w") as f:
            json.dump({
                "species": species,
                "sigma_eff": sigmai,
                "flag": flag,
                "volume_factors": [str(vf) for vf in volume_factors],
                "variables": [str(v) for v in flat_vars],
                "function": str(f_void_func),
                "expression": str(f_void),
            }, f, indent=4)
        print(f"✅ Void free energy exported: {out_file}")

    return result


# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."

    hc_data_example = {
        "species": ["A", "B"],
        "sigma": [1.0, 0.8],
        "flag": [1, 0]
    }

    out = free_energy_void(Ctx(), hc_data=hc_data_example)
    print("Species:", out["species"])
    print("Densities:", out["densities"])
    print("Volume factors:", out["volume_factors"])
    print("Flattened variables tuple:", out["variables"])
    print("Symbolic void free energy:\n", out["f_void"])

