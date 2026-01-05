import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.pyplot as plt

from cdft_solver.generators.potential.pair_potential_isotropic_registry import (
    register_isotropic_pair_potential
)
from cdft_solver.generators.potential_splitter.mf_potential_registry import (
    register_potential_converter
)


# -------------------- Potential factories --------------------

def supplied_potential_factory(r_data, U_data):
    """Return callable interpolated potential V(r)."""
    interp = interp1d(
        r_data,
        U_data,
        kind="linear",
        bounds_error=False,
        fill_value=(U_data[0], 0.0),
    )

    def V(r):
        r = np.asarray(r)
        return interp(r)

    return V


def wca_split(r, U):
    """Split a potential into repulsive (soft) and attractive parts (WCA)."""
    idx_min = np.argmin(U)
    r_min = r[idx_min]
    U_min = U[idx_min]

    U_rep = np.zeros_like(U)
    U_att = np.zeros_like(U)

    for i, ri in enumerate(r):
        if ri <= r_min:
            U_rep[i] = U[i] - U_min
            U_att[i] = U_min
        else:
            U_rep[i] = 0.0
            U_att[i] = U[i]

    return U_rep, U_att


def has_hard_core(U, threshold=1e3):
    """Heuristic to detect a hard-core in a potential."""
    return np.any(U > threshold)


# -------------------- Recursive helpers --------------------

def find_key_recursive(d, key):
    """Recursively find the first occurrence of a key in a nested dict."""
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


def replace_primary_potential_recursive(config, pair, new_name):
    """
    Recursively search for a 'primary' section inside 'potentials' and
    replace the type of the given pair with new_name.
    """
    if not isinstance(config, dict):
        return

    if "potentials" in config and isinstance(config["potentials"], dict):
        pots = config["potentials"]
        if "primary" in pots and isinstance(pots["primary"], dict):
            primary = pots["primary"]
            if pair in primary:
                primary[pair]["type"] = new_name

    # Recurse into nested dictionaries
    for v in config.values():
        if isinstance(v, dict):
            replace_primary_potential_recursive(v, pair, new_name)


# -------------------- Main registration wrapper --------------------

def register_supplied_potentials(ctx,
                                 config: dict,
                                 supplied_data: dict,
                                 export_json=True,
                                 export_plot=True):
    """
    Register supplied potentials as callable functions, split if needed,
    update the config, export JSON, and generate plots of full, soft, and
    attractive potentials.
    """
    potentials = find_key_recursive(supplied_data, "potentials")
    if potentials is None:
        return  # nothing to register

    out = Path(ctx.scratch_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = Path(ctx.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    # Store analysis data for JSON export
    potential_analysis = {}

    for family, family_data in potentials.items():
        for pair, pot in family_data.items():
            r = np.asarray(pot["r"])
            U = np.asarray(pot["U"])

            base_name = f"supplied_{pair}"

            # ---------- Full supplied potential ----------
            V_full = supplied_potential_factory(r, U)
            register_isotropic_pair_potential(base_name, lambda p, V=V_full: V)

            analysis_entry = {
                "r": r.tolist(),
                "U_full": U.tolist()
            }

            # ---------- Hard-core / WCA split ----------
            if has_hard_core(U):
                U_rep, U_att = wca_split(r, U)

                V_soft = supplied_potential_factory(r, U_rep)
                V_attr = supplied_potential_factory(r, U_att)

                # Soft repulsive potential
                register_isotropic_pair_potential(
                    f"{base_name}_soft", lambda p, V=V_soft: V
                )
                # Attractive potential
                register_isotropic_pair_potential(
                    f"{base_name}_attr", lambda p, V=V_attr: V
                )

                # MF uses attractive part only
                def supplied_to_mf(pot, name=base_name):
                    pot["type"] = f"{name}_attr"
                    return pot

                register_potential_converter(base_name, supplied_to_mf)

                analysis_entry["U_soft"] = U_rep.tolist()
                analysis_entry["U_attr"] = U_att.tolist()

            # ---------- Update config recursively ----------
            replace_primary_potential_recursive(config, pair, base_name)

            # ---------- Add to analysis dict ----------
            potential_analysis[pair] = analysis_entry

            # ---------- Generate plots ----------
            if export_plot:
                plt.figure()
                plt.plot(r, U, label="Full")
                if "U_soft" in analysis_entry:
                    plt.plot(r, analysis_entry["U_soft"], label="Repulsive")
                    plt.plot(r, analysis_entry["U_attr"], label="Attractive")
                plt.xlabel("r")
                plt.ylabel("U(r)")
                plt.title(f"Potential analysis: {pair}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots / f"potential_{pair}.png")
                plt.close()

    # ---------- Export JSON ----------
    if export_json:
        import json
        json_path = out / "supplied_potentials_analysis.json"
        with open(json_path, "w") as f:
            json.dump(potential_analysis, f, indent=2)

    return potential_analysis
