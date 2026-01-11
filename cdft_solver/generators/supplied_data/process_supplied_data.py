import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections.abc import Mapping
from mpl_toolkits.mplot3d import Axes3D  # REQUIRED for 3D plotting



def find_key_recursive(d, key):
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            out = find_key_recursive(v, key)
            if out is not None:
                return out
    return None




def _load_xy_file(path: Path):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        raise SuppliedDataError(f"Failed to read file '{path}': {e}")

    if data.ndim == 1:
        raise SuppliedDataError(
            f"File '{path}' must have at least 2 columns"
        )

    x = data[:, 0]
    y = data[:, 1:] if data.shape[1] > 2 else data[:, 1]

    return x, y



def materialize_supplied_data(obj, base_dir: Path):
    """
    Recursively walk supplied_data and replace
    { "filename": ... } blocks with loaded numerical data.
    """

    # Case 1: dictionary
    if isinstance(obj, dict):

        # File-backed leaf
        if "filename" in obj:
            filepath = base_dir / obj["filename"]

            if not filepath.exists():
                raise SuppliedDataError(f"File not found: {filepath}")

            x, y = _load_xy_file(filepath)

            # Replace filename with actual data
            new_obj = dict(obj)   # shallow copy
            del new_obj["filename"]
            new_obj["x"] = x
            new_obj["y"] = y

            return new_obj

        # Normal dictionary → recurse
        return {
            k: materialize_supplied_data(v, base_dir)
            for k, v in obj.items()
        }

    # Case 2: list → recurse elementwise
    if isinstance(obj, list):
        return [materialize_supplied_data(v, base_dir) for v in obj]

    # Case 3: scalar → unchanged
    return obj



def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _plot_materialized(obj, plots_dir: Path, prefix=""):
    if isinstance(obj, dict):

        # Leaf with x–y data
        if "x" in obj and "y" in obj:
            x = obj["x"]
            y = obj["y"]

            plt.figure()

            if isinstance(y, np.ndarray) and y.ndim == 2:
                for i in range(y.shape[1]):
                    plt.plot(x, y[:, i], label=f"y[{i}]")
                plt.legend()
            else:
                plt.plot(x, y)

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(prefix.rstrip("_"))
            plt.tight_layout()

            fname = prefix.rstrip("_") or "supplied_data"
            plt.savefig(plots_dir / f"{fname}.png")
            plt.close()
            return

        # Recurse
        for k, v in obj.items():
            _plot_materialized(v, plots_dir, prefix=f"{prefix}{k}_")

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _plot_materialized(v, plots_dir, prefix=f"{prefix}{i}_")



def process_supplied_data(
    ctx,
    config: dict,
    export_json=True,
    export_plot=True
):
    supplied_data = find_key_recursive(config, "supplied_data")

    if supplied_data is None:
        return None

    if not isinstance(supplied_data, dict):
        raise SuppliedDataError("'supplied_data' must be a dictionary")

    base_dir = Path(ctx.work_dir) if hasattr(ctx, "work_dir") else Path.cwd()

    # -----------------------------------------
    # Materialize filenames → data
    # -----------------------------------------
    materialized = materialize_supplied_data(supplied_data, base_dir)

    # -----------------------------------------
    # Export JSON
    # -----------------------------------------
    if export_json:
        out_dir = Path(ctx.scratch_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "supplied_data_materialized.json", "w") as f:
            json.dump(_to_serializable(materialized), f, indent=2)

    # -----------------------------------------
    # Export plots
    # -----------------------------------------
    if export_plot:
        plots_dir = Path(ctx.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_materialized(materialized, plots_dir)

    return materialized
