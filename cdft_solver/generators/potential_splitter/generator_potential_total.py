import json
from pathlib import Path


def raw_potentials(ctx):
    """
    Loads the raw particle interaction data from the given context (ctx)
    and returns the 'interactions' section of the JSON file directly.
    """
    scratch = Path(ctx.scratch_dir)
    input_file = Path(ctx.input_file)

    particle_json_in_scratch = scratch / "input_data_particles_interactions_parameters.json"
    particle_json_path = particle_json_in_scratch if particle_json_in_scratch.exists() else input_file

    if not particle_json_path.exists():
        raise FileNotFoundError(f"Particle interactions JSON not found: {particle_json_path}")

    # Load JSON data
    with open(particle_json_path, "r") as f:
        data = json.load(f)

    # Extract and return interactions
    interactions = data["particles_interactions_parameters"]["interactions"]
    return interactions


# Example usage (for testing only)
if __name__ == "__main__":
    class DummyCtx:
        scratch_dir = "."
        input_file = "interactions.json"

    ctx = DummyCtx()
    interactions = export_raw_potentials(ctx)
    print(json.dumps(interactions, indent=4))

