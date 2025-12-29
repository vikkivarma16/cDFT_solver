import json
import numpy as np
from pathlib import Path
from cdft_solver.generators.potential.generator_pair_potential_isotropic import (
    pair_potential_isotropic as ppi
)


def raw_potentials(
    ctx,
    grid_points=5000,
    filename_prefix="supplied_data_potential_raw_"
):
    """
    Load raw interaction data, ADD all level potentials for each pair,
    and export the resulting raw pair potentials.

    Rules
    -----
    • All levels (primary / secondary / tertiary) are summed
    • One output potential per pair
    • Output files: raw_potential_<pair>.txt
    """

    scratch = Path(ctx.scratch_dir)

    particle_json_in_scratch = scratch / "input_data_particles_interactions_parameters.json"
    particle_json_path = (
        particle_json_in_scratch
        if particle_json_in_scratch.exists()
        else Path(ctx.input_file)
    )

    if not particle_json_path.exists():
        raise FileNotFoundError(f"Particle interactions JSON not found: {particle_json_path}")

    with open(particle_json_path, "r") as f:
        data = json.load(f)

    interactions = data["particles_interactions_parameters"]["interactions"]
    levels = ["primary", "secondary", "tertiary"]

    # ---------------------------------------------------------
    # Collect interactions by pair across all levels
    # ---------------------------------------------------------
    pair_dict = {}

    for level in levels:
        for key, inter in interactions.get(level, {}).items():
            pair_dict.setdefault(key, []).append(inter)

    # ---------------------------------------------------------
    # Export summed raw potentials
    # ---------------------------------------------------------
    for key, inter_list in pair_dict.items():

        # Determine cutoff = max cutoff across levels
        cutoff = max(
            inter.get("cutoff", inter.get("sigma", 1.0) * 5.0)
            for inter in inter_list
        )

        r = np.linspace(1e-5, cutoff, grid_points)
        u_total = np.zeros_like(r)

        for inter in inter_list:
            V = ppi(inter)
            u_total += V(r)

        filename = scratch / f"{filename_prefix}{key}.txt"
        np.savetxt(
            filename,
            np.column_stack([r, u_total]),
            header="r U_raw(r)"
        )

        print(f"✅ Exported raw potential: {filename}")

