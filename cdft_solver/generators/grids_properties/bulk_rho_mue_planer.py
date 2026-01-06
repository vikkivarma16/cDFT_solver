import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def bulk_rho_mue_planer(
    ctx,
    thermodynamic_parameter: dict,
    r_space_coordinates: dict,
    export_json=True,
    filename="supplied_data_bulk_mu_rho_r_space.json",
    plot=False,
):
    """
    Assign bulk density and chemical potential values to an r-space grid
    based on planar phase coexistence along x.

    Parameters
    ----------
    ctx : object
        Must have `scratch_dir`
    thermodynamic_parameter : dict
        Output from coexistence finder:
        {
            "species": [...],
            "n_phases": int,
            "rhos_per_phase": [[...], ...],
            "mu_per_phase": [[...], ...],
            ...
        }
    r_space_coordinates : dict
        {
            "r_space": [[x,y,z], ...]
        }
    """

    # -------------------------
    # Validate ctx
    # -------------------------
    if ctx is None or not hasattr(ctx, "scratch_dir"):
        raise ValueError("ctx must provide scratch_dir")

    scratch_dir = Path(ctx.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Thermodynamic data
    # -------------------------
    species = thermodynamic_parameter["species"]
    n_species = len(species)
    n_phases = thermodynamic_parameter["n_phases"]

    rhos_per_phase = np.asarray(thermodynamic_parameter["rhos_per_phase"])
    mu_per_phase   = np.asarray(thermodynamic_parameter["mu_per_phase"])

    if rhos_per_phase.shape != (n_phases, n_species):
        raise ValueError("rhos_per_phase has inconsistent shape")

    if mu_per_phase.shape != (n_phases, n_species):
        raise ValueError("mu_per_phase has inconsistent shape")

    # Equal phase fractions unless specified elsewhere
    phase_fractions = np.ones(n_phases) / n_phases

    # -------------------------
    # r-space
    # -------------------------
    r_space = np.asarray(r_space_coordinates["r_space"])
    if r_space.ndim != 2 or r_space.shape[1] != 3:
        raise ValueError("r_space must have shape (N, 3)")

    n_points = r_space.shape[0]

    # -------------------------
    # Phase boundaries along x
    # -------------------------
    x_vals = r_space[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()

    cum_frac = np.cumsum(phase_fractions)
    cum_frac[-1] = 1.0
    boundaries = x_min + cum_frac * (x_max - x_min)

    # -------------------------
    # Assign phase per point
    # -------------------------
    phase_indices = np.zeros(n_points, dtype=int)

    for i, x in enumerate(x_vals):
        for p, b in enumerate(boundaries):
            if x <= b:
                phase_indices[i] = p
                break

    # -------------------------
    # Build bulk fields
    # -------------------------
    bulk_rhos = np.zeros((n_points, n_species))
    bulk_mues = np.zeros((n_points, n_species))

    for i in range(n_points):
        p = phase_indices[i]
        bulk_rhos[i] = rhos_per_phase[p]
        bulk_mues[i] = mu_per_phase[p]

    # -------------------------
    # Output dictionary
    # -------------------------
    result = {
        "r_space": r_space.tolist(),
        "species": species,
        "bulk_rhos": bulk_rhos.tolist(),
        "bulk_mues": bulk_mues.tolist(),
        "phase_indices": phase_indices.tolist(),
    }

    # -------------------------
    # Export JSON
    # -------------------------
    if export_json:
        out_file = scratch_dir / filename
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Bulk μ and ρ exported: {out_file}")

    # -------------------------
    # Optional plot
    # -------------------------
    if plot:
        plot_dir = scratch_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 4))
        for p in range(n_phases):
            idx = np.where(phase_indices == p)[0]
            plt.scatter(x_vals[idx], np.full(len(idx), p), s=8, label=f"Phase {p}")

        plt.xlabel("x")
        plt.ylabel("Phase index")
        plt.title("Planar phase assignment")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_file = plot_dir / "bulk_phase_assignment.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"✅ Phase assignment plot saved: {plot_file}")

    return result

