import numpy as np
import sympy as sp
import json
from pathlib import Path

def free_energy_EMF(ctx=None, hc_data=None, export_json=True, filename="Solution_EMF.json"):
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be provided as a dictionary")

    species = list(hc_data.get("species", []))
    sigma_raw = hc_data.get("sigma", [])
    flag_raw = hc_data.get("flag", [])

    n_species = len(species)
    if not species or sigma_raw is None or flag_raw is None:
        raise ValueError("hc_data must contain 'species', 'sigma', and 'flag'")
    if not (len(species) == len(sigma_raw) == len(flag_raw)):
        raise ValueError("Length mismatch between species, sigma, and flag")

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
    flag   = [int(f) for f in extract_diagonal(flag_raw, n_species, name="flag")]

    # symbolic densities
    densities = [sp.symbols(f"rho_{s}") for s in species]

    # symbolic interactions
    vij = [[sp.symbols(f"v_{species[i]}_{species[j]}") for j in range(n_species)]
           for i in range(n_species)]

    # volume correction factors
    volume_factors = []
    for j in range(n_species):
        factor = 0
        for i in range(n_species):
            if flag[i] == 1 and i != j:
                avg_p_vol = 0.5 * (sigmai[i]**3 + sigmai[j]**3)
                term = sp.Rational(1, 2) * densities[i] * (sp.pi / 6) * avg_p_vol - sp.Rational(3, 8) * (densities[i] * (sp.pi / 6) * avg_p_vol) ** 2
                factor += term
        volume_factors.append(1 - factor)

    # symbolic EMF free energy
    f_mf = sum(sp.Rational(1,2) * vij[i][j] * densities[i] * densities[j] / volume_factors[j]
               for i in range(n_species) for j in range(n_species))

    # flatten all variables for Lambda
    flat_vars = tuple(densities + [vij[i][j] for i in range(n_species) for j in range(n_species)])
    f_mf_func = sp.Lambda(flat_vars, f_mf)

    # prepare result
    result = {
        "variables": flat_vars,
        "function": f_mf_func,
        "expression": f_mf,
    }

    # optional JSON export
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
                "function": str(f_mf_func),
                "expression": str(f_mf),
            }, f, indent=4)
        print(f"✅ EMF free energy exported: {out_file}")

    return result


# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."

    hc_data_example = {
        "species": ["A", "B"],
        "sigma": [1.0, 0.8],
        "flag": [1, 0],
        "potentials": {}
    }

    out = free_energy_EMF(Ctx(), hc_data=hc_data_example)
    print("Species:", out["species"])
    print("Densities:", out["densities"])
    print("Volume factors:", out["volume_factors"])
    print("Flattened variables tuple:", out["variables"])
    print("Symbolic EMF free energy:\n", out["expression"])

