# cdft_solver/data_generators/solution_initiator.py

import json
from pathlib import Path


def data_exporter_solution_initiator(ctx):
    """
    Unified solution initiator exporter for isochemical and isochoric ensembles.

    Ensemble type is read from the input file as:
        ensemble = isochem   # or isochor

    Automatically selects the appropriate parser:
        - 'isochem' → chemical potentials + pressure
        - 'isochor' → species fractions + pressure/enthalpy
    """

    # -----------------------------
    # Read ensemble from input file
    # -----------------------------
    ensemble_type = "isochem"  # default
    input_file = Path(ctx.input_file)

    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            if line.lower().startswith("ensemble"):
                try:
                    _, value = line.split("=", 1)
                    ensemble_type = value.strip().lower()
                except Exception as e:
                    raise ValueError(f"Error parsing ensemble line: {line}\n{e}")
                break

    if ensemble_type not in ["isochem", "isochores"]:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    # -----------------------------
    # Call appropriate parser
    # -----------------------------
    if ensemble_type == "isochem":
        return _data_exporter_isochem(ctx, input_file, ensemble_type)
    else:
        return _data_exporter_isochor(ctx, input_file, ensemble_type)


# -----------------------------
# Isochemical parser
# -----------------------------
def _data_exporter_isochem(ctx, input_file, ensemble_type):
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_constraints = {}
    extrinsic_constraints = {
        "number_of_phases": None,
        "heterogeneous_pair": None,
        "total_density_bound": None,
    }

    chemical_potential = {}

    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()
            if not line or line.lower().startswith("ensemble"):
                continue

            if line.lower().startswith("chemical_potential"):
                try:
                    _, rest = line.split(":", 1)
                    species_part, value = rest.split("=")
                    species = species_part.strip()
                    value = float(value.strip())
                    chemical_potential[species] = value
                except Exception as e:
                    raise ValueError(f"Error parsing chemical_potential line: {line}\n{e}")
                continue

            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue

            key = key.strip().lower()
            value = value.strip()

            if key == "number_of_phases":
                extrinsic_constraints["number_of_phases"] = int(value)
            elif key == "heterogeneous_pair":
                extrinsic_constraints["heterogeneous_pair"] = value
            elif key == "total_density_bound":
                extrinsic_constraints["total_density_bound"] = float(value)
            elif key in ["pressure", "enthalpy"]:
                intrinsic_constraints[key] = float(value)

    if chemical_potential:
        intrinsic_constraints["chemical_potential"] = chemical_potential

    missing_extrinsic = [k for k, v in extrinsic_constraints.items() if v is None]
    if missing_extrinsic:
        raise ValueError(f"Missing required extrinsic parameters: {', '.join(missing_extrinsic)}")

    result = {
        "ensemble": ensemble_type,
        "solution_initiator": {
            "intrinsic_constraints": intrinsic_constraints,
            "extrinsic_constraints": extrinsic_constraints,
        },
    }

    output_file_json = output_dir / "input_data_solution_initiator.json"
    with open(output_file_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Isochemical solution initiator JSON exported to: {output_file_json}\n")
    return output_file_json


# -----------------------------
# Isochoric parser
# -----------------------------
def _data_exporter_isochor(ctx, input_file, ensemble_type):
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_constraints = {}
    extrinsic_constraints = {
        "number_of_phases": None,
        "heterogeneous_pair": None,
        "total_density_bound": None,
    }

    species_fraction = {}

    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()
            if not line or line.lower().startswith("ensemble"):
                continue

            if line.lower().startswith("species_fraction"):
                try:
                    _, rest = line.split(":", 1)
                    species_part, value = rest.split("=")
                    species = species_part.strip()
                    value = float(value.strip())
                    species_fraction[species] = value
                except Exception as e:
                    raise ValueError(f"Error parsing species_fraction line: {line}\n{e}")
                continue

            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue

            key = key.strip().lower()
            value = value.strip()

            if key == "number_of_phases":
                extrinsic_constraints["number_of_phases"] = int(value)
            elif key == "heterogeneous_pair":
                extrinsic_constraints["heterogeneous_pair"] = value
            elif key == "total_density_bound":
                extrinsic_constraints["total_density_bound"] = float(value)
            elif key in ["pressure", "enthalpy"]:
                intrinsic_constraints[key] = float(value)

    if species_fraction:
        intrinsic_constraints["species_fraction"] = species_fraction

    missing_extrinsic = [k for k, v in extrinsic_constraints.items() if v is None]
    if missing_extrinsic:
        raise ValueError(f"Missing required extrinsic parameters: {', '.join(missing_extrinsic)}")

    result = {
        "ensemble": ensemble_type,
        "solution_initiator": {
            "intrinsic_constraints": intrinsic_constraints,
            "extrinsic_constraints": extrinsic_constraints,
        },
    }

    output_file_json = output_dir / "input_data_solution_initiator.json"
    with open(output_file_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Isochoric solution initiator JSON exported to: {output_file_json}\n")
    return output_file_json


# -----------------------------
# Entry point
# -----------------------------
def main():
    import argparse
    from cdft_solver.utils import get_unique_dir

    parser = argparse.ArgumentParser(
        description="Export solution initiator parameters for isochem or isochor ensemble."
    )
    parser.add_argument("input_file", type=str, help="Path to input file containing solution parameters")
    args = parser.parse_args()

    scratch_dir = get_unique_dir("scratch")

    class Ctx:
        input_file = args.input_file
        scratch_dir = scratch_dir

    ctx = Ctx()
    data_exporter_solution_initiator(ctx)


if __name__ == "__main__":
    main()

