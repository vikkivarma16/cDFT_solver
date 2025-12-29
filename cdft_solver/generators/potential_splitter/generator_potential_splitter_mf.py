import json
import numpy as np
from pathlib import Path

from cdft_solver.generators.potential.generator_pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)


def meanfield_potentials(
    ctx,
    mode="meanfield",
    grid_points=5000,
    mf_json_name="potential_parameter_mf.json",
    filename_prefix="supplied_data_potential_mf_",
):
    """
    Loads particle interaction data, converts them for mean-field use,
    exports mean-field potentials to text files, and writes mf_parameter.json.

    Parameters
    ----------
    ctx : object
        Context object with 'scratch_dir' and optionally 'input_file'
    mode : str
        'meanfield' or 'raw'
    grid_points : int
        Grid size for exported potentials
    mf_json_name : str
        Output JSON filename
    filename_prefix : str
        Prefix for exported potential files
    """

    scratch = Path(ctx.scratch_dir)
    particle_json_in_scratch = scratch / "input_data_particles_interactions_parameters.json"
    particle_json_path = (
        particle_json_in_scratch if particle_json_in_scratch.exists()
        else getattr(ctx, "input_file", None)
    )

    if particle_json_path is None or not Path(particle_json_path).exists():
        raise FileNotFoundError(f"Particle interactions JSON not found in {scratch}")

    with open(particle_json_path) as f:
        data = json.load(f)

    species = data["particles_interactions_parameters"].get("species", [])
    interactions = data["particles_interactions_parameters"]["interactions"]

    # -------------------------------------------------------------
    # RAW MODE
    # -------------------------------------------------------------
    if mode.lower() == "raw":
        out = scratch / mf_json_name
        with open(out, "w") as f:
            json.dump(
                {"species": species, "interactions": interactions},
                f,
                indent=2
            )
        print(f"✅ Raw interactions written to {out}")
        return

    # -------------------------------------------------------------
    # Mean-field conversion rules
    # -------------------------------------------------------------
    def convert_potential(potential):
        ptype = potential["type"].lower()

        if ptype in ("hc", "ghc"):
            return {"type": "zero_potential"}

        elif ptype == "lj":
            new_pot = potential.copy()
            new_pot["type"] = "salj"
            return new_pot

        elif ptype == "mie":
            new_pot = potential.copy()
            new_pot["type"] = "ma"
            return new_pot

        return potential

    converted = {}
    all_hardcore = True

    for level, pairs in interactions.items():
        converted[level] = {}
        for key, potential in pairs.items():
            ptype = potential.get("type", "").lower()
            if ptype not in ("hc", "ghc"):
                all_hardcore = False
            converted[level][key] = convert_potential(potential)

    # -------------------------------------------------------------
    # Trivial case: no mean-field contribution
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    # Export mean-field potentials to text files
    # -------------------------------------------------------------
   # -------------------------------------------------------------
# Export TOTAL mean-field potentials (sum over all levels)
# -------------------------------------------------------------

# Collect all unique pair keys across all levels
    all_pairs = set()
    for pairs in converted.values():
        all_pairs.update(pairs.keys())

    for key in sorted(all_pairs):

        # Determine cutoff consistently (max over levels)
        cutoff = 0.0
        for level in converted:
            if key in converted[level]:
                inter = converted[level][key]
                cutoff = max(
                    cutoff,
                    inter.get("cutoff", inter.get("sigma", 1.0) * 5.0)
                )

        r = np.linspace(1e-5, cutoff, grid_points)
        u_total = np.zeros_like(r)

        # --- Sum contributions from all levels ---
        for level, pairs in converted.items():
            if key not in pairs:
                continue

            inter = pairs[key]
            ptype = inter.get("type", "").lower()

            if ptype == "zero_potential":
                continue

            V = ppi(inter)
            u_total += V(r)

        # --- Save summed potential ---
        fname = scratch / f"{filename_prefix}{key}.txt"
        np.savetxt(
            fname,
            np.column_stack([r, u_total]),
            header="r U_total(r)  # summed over all levels"
        )

        print(f"✅ Exported TOTAL mean-field potential: {fname}")


    # -------------------------------------------------------------
    # Write mf_parameter.json
    # -------------------------------------------------------------
    out = scratch / mf_json_name
    with open(out, "w") as f:
        json.dump(
            {"species": species, "interactions": converted},
            f,
            indent=2
        )

    print(f"✅ Mean-field parameters written to {out}")

