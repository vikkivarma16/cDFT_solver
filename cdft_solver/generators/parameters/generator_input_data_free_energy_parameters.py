import json
from pathlib import Path


def data_exporter_free_energy_parameters(ctx, result: dict = None):
    """
    Export free-energy configuration parameters to JSON and a readable TXT file.

    Input file example (expected format):

        # choose mode for the calculation of free energy standard or advanced
        # where standard will use perturbative mean field while advanced use FMT integrated mean field
        mode = standard

        # choose method for the calculation of mean-field free energy
        # it can be smf, emf or cavity
        method = smf

        # choose mode for the calculation of integrated strength
        # it can be rdf, uniform or void
        # where rdf and void can have supplied data
        # uniform does not need any data
        integrated_strength_kernel = rdf
        supplied_data = no
    """

    input_file = Path(ctx.input_file)
    output_dir = Path(ctx.scratch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize parameters ---
    mode = None
    method = None
    integrated_strength_kernel = None
    supplied_data = None

    # --- Parse input parameters from file ---
    with open(input_file, "r") as file:
        for raw_line in file:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue

            key = key.strip().lower()
            value = value.strip().lower()

            if key == "mode":
                mode = value
            elif key == "method":
                # Disambiguate by checking if already used for mean-field or strength
                if method is None:
                    method = value
                else:
                    integrated_strength_kernel = value
            elif key in ("integrated_strength_kernel", "integrated_strength_kernel"):
                integrated_strength_kernel = value
            elif key in ("supplied_data", "supplied"):
                supplied_data = value

    # --- Validation ---
    missing = []
    if mode is None:
        missing.append("mode")
    if method is None:
        missing.append("method")
    if integrated_strength_kernel is None:
        missing.append("integrated_strength_kernel")
    if supplied_data is None:
        missing.append("supplied_data")

    if missing:
        raise ValueError(f"Missing required parameters in input file: {', '.join(missing)}")

    # --- Construct dictionary for export ---
    parameters = {
        "mode": mode,
        "method": method,
        "integrated_strength_kernel": integrated_strength_kernel,
        "supplied_data": supplied_data,
    }

    if result is not None:
        parameters["result_summary"] = result

    # --- Export JSON ---
    output_file_json = output_dir / "input_data_free_energy_parameters.json"
    with open(output_file_json, "w") as f:
        json.dump({"free_energy_parameters": parameters}, f, indent=4)

    print(f"\n✅ Free-energy parameters JSON exported to: {output_file_json}")


    return output_file_json

