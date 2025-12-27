# cdft_solver/data_generators/solution_initiator.py

import json
from pathlib import Path


def data_exporter_iso_rdf_parameters(ctx):
    """
    Reads RDF-related parameters (densities, r_min, r_max, n_points, beta, alpha_rdf_max)
    from an input file and exports them to JSON for isotropic RDF computation.

    Expected input format:
        # --- densities in the system ---
        density: a = 0.5
        #density: b = 0.02
        #density: c = 0.09

        # --- RDF parameters ---
        cutoff = 7
        n_points = 400
        beta = 1.0
        alpha_rdf_max = 0.1

    Output JSON structure:
    {
        "densities": {
            "a": 0.5
        },
        "rdf_parameters": {
            "r_min": 0.0175,
            "r_max": 7.0,
            "n_points": 400,
            "beta": 1.0,
            "alpha_rdf_max": 0.1
        }
    }
    """

    # -----------------------------
    # Prepare paths
    # -----------------------------
    input_file = Path(ctx.input_file)
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    densities = {}
    rdf_parameters = {}

    # -----------------------------
    # Parse input file
    # -----------------------------
    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()  # remove comments
            if not line:
                continue

            # --- Parse densities ---
            if line.lower().startswith("density"):
                try:
                    _, rest = line.split(":", 1)
                    species_part, value = rest.split("=")
                    species = species_part.strip()
                    value = float(value.strip())
                    densities[species] = value
                except Exception as e:
                    raise ValueError(f"Error parsing density line: {line}\n{e}")
                continue

            # --- Parse RDF parameters ---
            if any(line.lower().startswith(p) for p in ["cutoff", "n_points", "beta", "alpha_rdf_max"]):
                try:
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    rdf_parameters[key] = int(value) if key == "n_points" else float(value)
                except Exception as e:
                    raise ValueError(f"Error parsing RDF parameter line: {line}\n{e}")

    # -----------------------------
    # Validation and derived params
    # -----------------------------
    if not densities:
        raise ValueError("No density values found in input file.")

    required_rdf_keys = ["cutoff", "n_points", "beta", "alpha_rdf_max"]
    missing_rdf = [k for k in required_rdf_keys if k not in rdf_parameters]
    if missing_rdf:
        raise ValueError(f"Missing RDF parameters: {', '.join(missing_rdf)}")

    # Derive r_min and r_max
    r_max = float(rdf_parameters["cutoff"])
    n_points = int(rdf_parameters["n_points"])
    r_min = r_max / n_points  # derived definition

    # Final structure
    rdf_parameters_final = {
        "r_min": r_min,
        "r_max": r_max,
        "n_points": n_points,
        "beta": float(rdf_parameters["beta"]),
        "alpha_rdf_max": float(rdf_parameters["alpha_rdf_max"]),
    }

    result = {
        "densities": densities,
        "rdf_parameters": rdf_parameters_final,
    }

    # -----------------------------
    # Write output JSON
    # -----------------------------
    output_file_json = scratch / "input_data_iso_rdf.json"
    with open(output_file_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ RDF parameter JSON exported to: {output_file_json}\n")
    return output_file_json


# -----------------------------
# Entry point
# -----------------------------
def main():
    import argparse
    from cdft_solver.utils import get_unique_dir

    parser = argparse.ArgumentParser(
        description="Export density and RDF parameters for isotropic RDF calculation."
    )
    parser.add_argument("input_file", type=str, help="Path to input file containing RDF parameters")
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Directory where RDF input JSON should be exported (default: auto-generated RDF dir)",
    )
    args = parser.parse_args()

    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else get_unique_dir("rdf_input")

    class Ctx:
        input_file = args.input_file
        scratch_dir = scratch_dir

    ctx = Ctx()
    data_exporter_iso_rdf_parameters(ctx)


if __name__ == "__main__":
    main()

