import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections.abc import Mapping
from mpl_toolkits.mplot3d import Axes3D  # REQUIRED for 3D plotting


class SuppliedDataError(RuntimeError):
    pass


def _load_four_column_density_file(path: Path):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        raise SuppliedDataError(f"Failed to read density file '{path}': {e}")

    if data.ndim != 2 or data.shape[1] != 4:
        raise SuppliedDataError(
            f"Density file '{path}' must have exactly 4 columns "
            f"(got shape {data.shape})"
        )

    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def _load_two_column_file(path: Path, label: str):
    try:
        data = np.loadtxt(path)
    except Exception as e:
        raise SuppliedDataError(f"Failed to read {label} file '{path}': {e}")

    if data.ndim != 2 or data.shape[1] != 2:
        raise SuppliedDataError(
            f"{label} file '{path}' must have exactly 2 columns (got shape {data.shape})"
        )

    return data[:, 0], data[:, 1]


def _recursive_find(section: dict, section_name: str, base_dir: Path):
    result = {}

    for key, value in section.items():
        if isinstance(value, dict) and "filename" not in value:
            result[key] = _recursive_find(value, section_name, base_dir)
            continue

        if "filename" not in value:
            raise SuppliedDataError(
                f"Missing 'filename' entry for {section_name}:{key}"
            )

        filepath = base_dir / value["filename"]

        if not filepath.exists():
            raise SuppliedDataError(f"File not found: {filepath}")

        if section_name == "potentials":
            r, U = _load_two_column_file(filepath, "potential")
            result[key] = {"r": r, "U": U}

        elif section_name == "rdf":
            r, gr = _load_two_column_file(filepath, "rdf")
            result[key] = {"r": r, "gr": gr}

        elif section_name == "densities":
            x, y, z, rho = _load_four_column_density_file(filepath)
            result[key] = {"x": x, "y": y, "z": z, "rho": rho}

        else:
            raise SuppliedDataError(f"Unknown supplied-data section '{section_name}'")

    return result


def process_supplied_data_inner(ctx, supplied_data: dict, export_json=False, export_plot=False):
    out = Path(ctx.scratch_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots = Path(ctx.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    base_dir = Path(ctx.work_dir) if hasattr(ctx, "work_dir") else Path.cwd()

    processed = {}

    for section_name in ("potentials", "rdf", "densities"):
        if section_name in supplied_data:
            processed[section_name] = _recursive_find(
                supplied_data[section_name],
                section_name,
                base_dir,
            )

    if export_json:
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        with open(out / "supplied_data_processed.json", "w") as f:
            json.dump(to_serializable(processed), f, indent=2)

    def plot_section(section, name):
        for key, val in section.items():
            if isinstance(val, dict) and not {"r", "x"}.intersection(val.keys()):
                plot_section(val, name)
                continue

            if name == "potentials":
                plt.figure()
                plt.plot(val["r"], val["U"])
                plt.xlabel("r")
                plt.ylabel("U(r)")
                plt.title(f"potential: {key}")
                plt.tight_layout()
                plt.savefig(plots / f"potentials_{key}.png")
                plt.close()

            elif name == "rdf":
                plt.figure()
                plt.plot(val["r"], val["gr"])
                plt.xlabel("r")
                plt.ylabel("g(r)")
                plt.title(f"rdf: {key}")
                plt.tight_layout()
                plt.savefig(plots / f"rdf_{key}.png")
                plt.close()

            elif name == "densities":
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                sc = ax.scatter(
                    val["x"], val["y"], val["z"],
                    c=val["rho"], s=5, cmap="viridis"
                )

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_title(f"density: {key}")
                fig.colorbar(sc, ax=ax, label="ρ")

                plt.tight_layout()
                plt.savefig(plots / f"densities_{key}.png")
                plt.close()

    if export_plot:
        for name, section in processed.items():
            plot_section(section, name)

    return processed



'''
def process_supplied_data(ctx, config: dict, export_json=False, export_plot=False):
    supplied_data = find_key_recursive(config, "supplied_data")

    if supplied_data is None:
        return None

    if not isinstance(supplied_data, dict):
        raise SuppliedDataError("'supplied_data' must be a dictionary")

    return process_supplied_data_inner(
        ctx=ctx,
        supplied_data=supplied_data,
        export_json=export_json,
        export_plot=export_plot,
    )
'''


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





