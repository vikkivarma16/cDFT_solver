import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def bulk_rho_mue_z_space_box(ctx, data_dict=None, r_space_coordinates=None, export_json=True,
                             filename="supplied_data_bulk_mue_rho_r_space.json", plot=False):
    """
    Assign bulk density and chemical potential values to an r-space grid.

    Parameters
    ----------
    ctx : object
        Must have `scratch_dir` for exporting files.
    data_dict : dict
        Dictionary containing nested keys:
            - "thermodynamics": {species, n_phases, rhos_per_phase, mu_per_phase, phase_fractions (optional)}
            - "r_space": Nx3 array [[x,y,z],...]
    r_space_coordinates : np.ndarray, optional
        Nx3 array, if already extracted from dictionary.
    export_json : bool
        If True, export output to JSON in ctx.scratch_dir
    filename : str
        Name of JSON output file
    plot : bool
        If True, plot phase assignment along x

    Returns
    -------
    dict
        {
            "r_space": [[x,y,z],...],
            "species": [...],
            "bulk_rhos": [...],
            "bulk_mues": [...],
            "phase_indices": [...]
        }
    """

    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx.scratch_dir must be provided")
    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    if data_dict is None:
        raise ValueError("data_dict must be provided")

    # -------------------------
    # Recursive search helper
    # -------------------------
    def find_key_recursive(d, key):
        """Recursively search for a key in nested dictionaries."""
        if not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        return None

    # --- Extract thermodynamic info ---
    thermo = find_key_recursive(data_dict, "thermodynamics")
    if thermo is None:
        raise KeyError("No 'thermodynamics' key found in data_dict")

    species_names = thermo["species"]
    n_species = len(species_names)
    n_phases = thermo["n_phases"]

    rhos_per_phase = np.array(thermo["rhos_per_phase"])
    mues_per_phase = np.array(thermo["mu_per_phase"])
    phase_fractions = np.array(thermo.get("phase_fractions", np.ones(n_phases) / n_phases))

    # --- Extract r-space ---
    r_space = r_space_coordinates
    if r_space is None:
        r_space = find_key_recursive(data_dict, "r_space")
        if r_space is None:
            raise KeyError("No 'r_space' key found in data_dict")
    r_space = np.array(r_space, dtype=float)
    n_points = r_space.shape[0]

    # --- Compute phase boundaries along x ---
    x_vals = r_space[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    cum_fractions = np.cumsum(phase_fractions)
    cum_fractions[-1] = 1.0
    boundaries = x_min + cum_fractions * (x_max - x_min)

    # --- Assign phase to each point ---
    phase_indices = np.zeros(n_points, dtype=int)
    for i, x in enumerate(x_vals):
        for p, b in enumerate(boundaries):
            if x <= b:
                phase_indices[i] = p
                break

    # --- Build output data dictionary ---
    bulk_rhos = []
    bulk_mues = []
    for i in range(n_points):
        p = phase_indices[i]
        bulk_rhos.append(list(rhos_per_phase[p]))
        bulk_mues.append(list(mues_per_phase[p]))

    result = {
        "r_space": r_space.tolist(),
        "species": species_names,
        "bulk_rhos": bulk_rhos,
        "bulk_mues": bulk_mues,
        "phase_indices": phase_indices.tolist()
    }

    # --- Export JSON ---
    if export_json:
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Bulk μ and ρ exported to {out_file}")

    # --- Optional plot ---
   # --- Optional plot ---
    if plot and ctx is not None:
        scratch_dir = Path(ctx.scratch_dir)
        plot_dir = scratch_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 4))
        for p in range(n_phases):
            idx = np.where(phase_indices == p)[0]
            plt.scatter(x_vals[idx], [p] * len(idx), s=10, label=f"Phase {p}")
        plt.xlabel("x")
        plt.ylabel("Phase index")
        plt.title("Phase Assignment along x")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_file = plot_dir / "bulk_phase_assignment.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"✅ Phase assignment plot saved: {plot_file}")


    return result

