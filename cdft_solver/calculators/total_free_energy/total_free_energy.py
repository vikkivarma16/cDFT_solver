import numpy as np
import sympy as sp

from cdft_solver.calculators.free_energy_ideal.ideal import ideal as idl
from cdft_solver.calculators.free_energy_mean_field.mean_field import mean_field
from cdft_solver.calculators.free_energy_hard_core.hard_core import hard_core
from cdft_solver.calculators.free_energy_hybrid.hybrid import hybrid
from cdft_solver.calculators.free_energy_hybrid_lattice.free_energy_lattice import free_energy_lattice



def unify_symbols(components):
    """
    Unify SymPy symbols across multiple free-energy components
    based on symbol *name* (not object identity).
    """

    canonical = {}   # name -> Symbol
    unified_components = []

    for comp in components:
        expr = comp["expression"]
        vars_ = comp["variables"]

        subs = {}

        for v in vars_:
            if v.name not in canonical:
                canonical[v.name] = v
            subs[v] = canonical[v.name]

        expr = expr.subs(subs)

        unified_components.append({
            **comp,
            "expression": expr,
            "variables": tuple(canonical[v.name] for v in vars_),
        })

    return unified_components, list(canonical.values())





def extract_sigma_matrix(hc_data):
    sigma_raw = hc_data.get("sigma", None)
    species = hc_data.get("species", [])

    if sigma_raw is None:
        return None

    n = len(species)
    arr = np.asarray(sigma_raw, dtype=float)

    if arr.ndim == 1 and arr.size == n:
        return np.diag(arr)

    if arr.ndim == 1 and arr.size == n * n:
        return arr.reshape((n, n))

    if arr.ndim == 2 and arr.shape == (n, n):
        return arr

    raise ValueError("Invalid sigma format")



def sigma_is_zero(sigma_matrix, tol=0.0):
    return sigma_matrix is None or np.all(np.abs(sigma_matrix) <= tol)

def merge_free_energies(components):
    """
    Merge multiple free-energy components into one symbolic object,
    unifying symbols by name across modules.
    """

    # ------------------------------------------------------------
    # Canonicalize symbols
    # ------------------------------------------------------------
    unified_components, unified_vars = unify_symbols(components)

    # ------------------------------------------------------------
    # Sum expressions
    # ------------------------------------------------------------
    total_expr = sp.Integer(0)
    for comp in unified_components:
        total_expr += comp["expression"]

    # ------------------------------------------------------------
    # Construct Lambda
    # ------------------------------------------------------------
    F_total = sp.Lambda(tuple(unified_vars), total_expr)

    return {
        "variables": tuple(unified_vars),
        "expression": total_expr,
        "function": F_total,
        "components": unified_components,
    }





def total_free_energy(
    ctx=None,
    hc_data=None,
    system_config=None,
    export_json=True,
    filenames = None
):
    """
    Unified free-energy dispatcher.

    Always includes:
      - Ideal free energy

    Conditional:
      - Mean-field only (sigma = 0)
      - Mean-field + hard-core (sigma != 0, mode = standard)
      - Hybrid (sigma != 0, mode = hybrid)
    """

    # -------------------------
    # Validate input
    # -------------------------
    if hc_data is None or not isinstance(hc_data, dict):
        raise ValueError("hc_data must be a dictionary")

    if system_config is None or "system" not in system_config:
        raise ValueError("system_config must contain 'system'")

    mode = system_config["system"].get("mode", "standard").lower()

    # -------------------------
    # Sigma inspection
    # -------------------------
    sigma_matrix = extract_sigma_matrix(hc_data)
    no_hard_core = sigma_is_zero(sigma_matrix)

    components = []

    # ============================================================
    # IDEAL FREE ENERGY (ALWAYS)
    # ============================================================
    ideal = idl(
        ctx=ctx,
        hc_data=hc_data,
        export_json=export_json,
        filename  = filenames["ideal"],   # densities reused
    )
    components.append(ideal)

    # ============================================================
    # PURE IDEAL + MEAN FIELD
    # ============================================================
    if no_hard_core:
        mf = mean_field(
            ctx=ctx,
            hc_data=hc_data,
            system_config=system_config,
            export_json=export_json,
            filename  = filenames["mean_field"]
        )
        components.append(mf)

        merged = merge_free_energies(components)
        merged["selected_model"] = "ideal + mean_field"
        return merged

    # ============================================================
    # STANDARD → IDEAL + MF + HC
    # ============================================================
    if mode == "standard":
        mf = mean_field(
            ctx=ctx,
            hc_data=hc_data,
            system_config=system_config,
            export_json=export_json,
            filename = filenames["mean_field"]
        )

        hc = hard_core(
            ctx=ctx,
            hc_data=hc_data,
            export_json=export_json,
            filename = filenames["hard_core"] 
        )

        components.extend([mf, hc])

        merged = merge_free_energies(components)
        merged["selected_model"] = "ideal + mean_field + hard_core"
        return merged

    # ============================================================
    # HYBRID → IDEAL + HYBRID
    # ============================================================
    if mode == "hybrid":
        hyb = hybrid(
            ctx=ctx,
            hc_data=hc_data,
            export_json=export_json,
            filename = filenames["hybrid"],
        )

        components.extend(hyb)

        merged = merge_free_energies(components)
        merged["selected_model"] = "ideal + hybrid"
        return merged

    if mode == "hybrid_lattice":
        hyb_lattice = free_energy_lattice(
            ctx=ctx,
            hc_data=hc_data,
            export_json=export_json,
            filename = filenames["hybrid"],
        )

        components.extend(hyb_lattice)

        merged = merge_free_energies(components)
        merged["selected_model"] = "ideal + hybrid_lattice"
        return merged
    # ============================================================
    # Unsupported mode
    # ============================================================
    raise ValueError(f"Unsupported system mode '{mode}'")
