# cdft_solver/executor_dft_main.py (UPDATED WITH profile HANDLING)

from pathlib import Path
import sys
from .utils import get_unique_dir, ExecutionContext

# Import executors
from .engines.bulk import bulk_executor as bulk_exec
from .engines.rdf_bulk import rdf_bulk_executor as rbe
from .engines.one_d_profile import one_d_profile_executor as odp_exec
# (OPTIONAL) future executors for 2D/3D profiles could be added here
# from .engines.two_d_profile import two_d_profile_executor as tdp_exec
# from .engines.three_d_profile import three_d_profile_executor as thdp_exec


def parse_input_keywords(input_text):
    """
    Parse ensemble, method, task, and profile from an input file.
    Supports both ':' and '=' formats.
    Lines with '#' comments are ignored.
    """
    params = {
        "ensemble": None,
        "method": None,
        "task": None,
        "profile": None,
    }

    for line in input_text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue

        if ":" in line:
            parts = line.split(":", 1)
        elif "=" in line:
            parts = line.split("=", 1)
        else:
            continue

        key = parts[0].strip().lower()
        value = parts[1].strip().lower()

        if key in params:
            params[key] = value

    return params


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m cdft_solver.executor_dft_main <executor_input.in>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    root = input_file.parent

    scratch = get_unique_dir("scratch", root=root)
    plots = get_unique_dir("plots", root=root)

    input_data = input_file.read_text()
    params = parse_input_keywords(input_data)

    ensemble = params["ensemble"]
    method = params["method"]
    task = params["task"]
   

    if task not in ["bulk", "inhomogeneous", "rdf_bulk"]:
        raise ValueError(f"Invalid task: {task}. Choose 'bulk', 'inhomogeneous', or 'rdf_bulk'.")

    ctx = ExecutionContext(
        input_file=input_file,
        input_data=input_data,
        scratch_dir=scratch,
        plots_dir=plots,
    )

    # -------------------------- DISPATCH LOGIC --------------------------
    if task == "bulk":
        if ensemble not in ["isochores", "isochem"]:
            raise ValueError(f"Unsupported ensemble: {ensemble}. Must be 'isochores' or 'isochem'.")
        if method not in ["emf", "smf", "void"]:
            raise ValueError(f"Invalid method: {method}. Choose 'emf', 'smf', or 'void'.")
        result = bulk_exec(ctx)

    elif task == "inhomogeneous":
        if ensemble not in ["isochores", "isochem"]:
            raise ValueError(f"Unsupported ensemble: {ensemble}. Must be 'isochores' or 'isochem'.")
        if method not in ["emf", "smf", "void"]:
            raise ValueError(f"Invalid method: {method}. Choose 'emf', 'smf', or 'void'.")

        # PROFILE HANDLING EXTENSION
        profile = params["profile"]
        if profile is None:
            raise ValueError("For task='inhomogeneous', a 'profile' must be specified.")

        if profile == "one_d":
            result = odp_exec(ctx)
        else:
            raise ValueError(f"Unsupported profile: {profile}. Only 'one_d' is implemented.")

    elif task == "rdf_bulk":
        result = rbe(ctx)

    else:
        raise ValueError(f"Task '{task}' is not supported.")

    print(f"\u2705 Execution completed for ensemble={ensemble}, method={method}, task={task}, profile={profile}")
    return result


if __name__ == "__main__":
    main()

