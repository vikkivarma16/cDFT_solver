# wall_potentials.py — evaluate and export wall-particle potentials on supplied r-space
EPSILON = 1e-6

def wall_potential_values(ctx):
    import numpy as np
    import json
    from pathlib import Path
    from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi

    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load confinement JSON and wall interactions
    # -----------------------------
    json_file_path = scratch / "input_data_space_confinement_parameters.json"
    with open(json_file_path, "r") as f:
        conf = json.load(f)

    walls_props = conf["space_confinement_parameters"].get("walls_properties", {})
    walls_positions = walls_props.get("walls_position", [])
    wall_particle_type = walls_props.get("walls_particles_type", [])

    interactions = walls_props.get("walls_interactions", {})
    primary = interactions.get("primary", {})
    secondary = interactions.get("secondary", {})
    tertiary = interactions.get("tertiary", {})

    # -----------------------------
    # Load species list
    # -----------------------------
    pdata_file = scratch / "input_data_particles_interactions_parameters.json"
    with open(pdata_file, "r") as f:
        pdata = json.load(f)
    species = pdata["particles_interactions_parameters"]["species"]

    # -----------------------------
    # Load supplied r-space / positions
    # -----------------------------
    pos_file = scratch / "supplied_data_r_space.txt"
    positions = np.loadtxt(pos_file)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    xs = positions[:, 0]

    # -----------------------------
    # Helper: compute V(r) using isotropic potential factory
    # -----------------------------
    def compute_potential(r, model_dict):
        V_func = ppi(model_dict)
        return V_func(r)

    # -----------------------------
    # Compute total external potential per species
    # -----------------------------
    v_ext_species = {}
    for spc in species:
        v_accum = np.zeros_like(xs)
        for wi, wpos in enumerate(walls_positions):
            ref = np.array(wpos)
            r_space = np.linalg.norm(positions - ref, axis=1)

            # wall particles at this wall
            if isinstance(wall_particle_type, list):
                wall_particles_here = wall_particle_type
            else:
                wall_particles_here = [wall_particle_type]

            for wp in wall_particles_here:
                pair1 = wp + spc
                pair2 = spc + wp

                # primary
                key = pair1 if pair1 in primary else (pair2 if pair2 in primary else None)
                if key:
                    v_accum += compute_potential(r_space, primary[key])
                # secondary
                key = pair1 if pair1 in secondary else (pair2 if pair2 in secondary else None)
                if key:
                    v_accum += compute_potential(r_space, secondary[key])
                # tertiary
                key = pair1 if pair1 in tertiary else (pair2 if pair2 in tertiary else None)
                if key:
                    v_accum += compute_potential(r_space, tertiary[key])

        v_ext_species[spc] = v_accum

    # -----------------------------
    # Export per-species potential along r-space
    # -----------------------------
    for spc, vvals in v_ext_species.items():
        outfile = scratch / f"supplied_data_walls_potential_{spc}_r_space.txt"
        with open(outfile, "w") as f:
            for pos, v in zip(positions, vvals):
                f.write(" ".join(map(str, pos)) + f" {v}\n")

    print("\n... external wall potentials exported successfully ...\n")
    return 0

# If run as script
if __name__ == "__main__":
    from types import SimpleNamespace
    ctx = SimpleNamespace(scratch_dir=".", plots_dir=".")
    wall_potential_values(ctx)

