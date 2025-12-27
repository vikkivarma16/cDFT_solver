import json
from pathlib import Path


def meanfield_potentials(ctx, mode="meanfield"):
    """
    Loads particle interaction data and returns a dictionary of potentials
    and species, optionally converting them for mean-field use.

    Parameters
    ----------
    ctx : object
        Context object with at least 'scratch_dir' attribute, and optionally 'input_file'.
    mode : str, optional
        'meanfield' → convert hc/ghc → zero_potential, lj→salj, mie→ma
        'raw'       → return potentials exactly as defined
        (default = 'meanfield')

    Returns
    -------
    dict
        {
            "species": [...],
            "interactions": {...} or None if all are hard-core
        }
    """

    scratch = Path(ctx.scratch_dir)
    particle_json_in_scratch = scratch / "input_data_particles_interactions_parameters.json"
    particle_json_path = (
        particle_json_in_scratch if particle_json_in_scratch.exists() else getattr(ctx, "input_file", None)
    )

    if particle_json_path is None or not Path(particle_json_path).exists():
        raise FileNotFoundError(f"Particle interactions JSON not found in {scratch} or fallback path.")

    with open(particle_json_path, "r") as f:
        data = json.load(f)

    # Load species and interactions
    species = data["particles_interactions_parameters"].get("species", [])
    interactions = data["particles_interactions_parameters"]["interactions"]

    # If raw mode, just return as-is
    if mode.lower() == "raw":
        return {"species": species, "interactions": interactions}

    # Otherwise, build mean-field-ready potentials
    def convert_potential(potential):
        """Convert potential definition for mean-field treatment."""
        ptype = potential["type"].lower()

        # Hard-core or excluded-volume → zero potential
        if ptype in ("hc", "ghc"):
            return {"type": "zero_potential"}

        # Lennard-Jones → split attractive LJ
        elif ptype == "lj":
            new_pot = potential.copy()
            new_pot["type"] = "salj"
            return new_pot

        # Mie → split attractive Mie
        elif ptype == "mie":
            new_pot = potential.copy()
            new_pot["type"] = "ma"
            return new_pot

        # Otherwise, keep original
        return potential

    # Apply conversion for all levels and pairs
    converted = {}
    all_hardcore = True  # flag to check if everything is hc/ghc

    for level, pairs in interactions.items():
        converted[level] = {}
        for pair, potential in pairs.items():
            ptype = potential.get("type", "").lower()
            if ptype not in ("hc", "ghc"):
                all_hardcore = False
            converted[level][pair] = convert_potential(potential)

    # If all interactions are just hard-core types → return None for mean-field interactions
    if all_hardcore:
        return {"species": species, "interactions": None}

    return {"species": species, "interactions": converted}


# Example usage
if __name__ == "__main__":
    class Ctx:
        scratch_dir = "."
        input_file = "interactions.json"

    data_out = meanfield_potentials(Ctx(), mode="meanfield")
    print(json.dumps(data_out, indent=4))

