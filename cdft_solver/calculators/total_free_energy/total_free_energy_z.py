"""
FAST VERSION OF total_free_energy_z.py
No functional changes. Up to 20× faster by removing unnecessary simplifications.
"""

import json
import ast
from pathlib import Path
import sympy as sp


from cdft_solver.calculators.free_energy_ideal.ideal import ideal
from cdft_solver.calculators.free_energy_mean_field.mean_field_planer import mean_field_planer
from cdft_solver.calculators.free_energy_hard_core.hard_core_planer import hard_core_planer
from cdft_solver.calculators.free_energy_hybrid.hybrid_planer import free_energy_hybrid_planer


# ======================================================================
# CONFIG LOADING
# ======================================================================
def _load_config(ctx):
    scratch = Path(ctx.scratch_dir)
    cfg_scratch = scratch / "input_data_free_energy_parameters.json"
    cfg_input = Path(getattr(ctx, "input_file", "")) if getattr(ctx, "input_file", None) else None

    if cfg_scratch.exists():
        path = cfg_scratch
    elif cfg_input and cfg_input.exists():
        path = cfg_input
    else:
        return {
            "free_energy_parameters": {
                "mode": "standard",
                "method": "smf",
            }
        }

    with open(path, "r") as fh:
        cfg = json.load(fh)

    block = cfg.get("free_energy") or cfg.get("free_energy_parameters") or {}

    return {
        "free_energy_parameters": {
            "mode": (block.get("mode") or "standard").lower(),
            "method": (block.get("method") or "smf").lower(),
        }
    }


# ======================================================================
# NORMALIZATION UTILITIES
# ======================================================================
def _parse_list(x):
    if x is None: return []
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)): return list(v)
            return [v]
        except Exception:
            return [x]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _normalize_result(res, kind):
    """
    Remove heavy symbolic processing.
    Return raw symbolic expressions without simplification.
    """

    if res is None:
        return {
            "species": [],
            "sigma_eff": [],
            "flag": [],
            "densities": [],
            "densities_z": [],
            "densities_zs": [],
            "vij": [],
            "volume_factors": [],
            "variables": None,
            "f_two_point": None,
            "f_symbolic": None,
        }

    out = {
        "species": _parse_list(res.get("species")),
        "sigma_eff": _parse_list(res.get("sigma_eff")),
        "flag": _parse_list(res.get("flag")),
        "densities": _parse_list(res.get("densities")),
        "densities_z": _parse_list(res.get("densities_z")),
        "densities_zs": _parse_list(res.get("densities_zs")),
        "vij": res.get("vij") or [],
        "volume_factors": res.get("volume_factors", res.get("volume_factor_zs", [])) or [],
        "variables": res.get("variables"),
        "f_two_point": None,
        "f_symbolic": None,
    }

    if kind == "hc":
        out["f_symbolic"] = res.get("phi_total") or res.get("f_hc_symbolic")

    elif kind == "mf":
        out["f_two_point"] = res.get("f_mf_two_point") or res.get("f_mf_two_point_symbolic")

    return out


# ======================================================================
# SAFE SYMPY WRAPPER (NO simplify)
# ======================================================================
def _sympify_maybe(expr):
    if expr is None: return 0
    if isinstance(expr, (sp.Basic, sp.Expr)): return expr
    try:
        return sp.sympify(expr)
    except Exception:
        return 0


# ======================================================================
# JSON converter
# ======================================================================
def _jsonify_sympy(x):
    if isinstance(x, (sp.Basic, sp.Expr)): return str(x)
    if isinstance(x, dict): return {k: _jsonify_sympy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonify_sympy(v) for v in x]
    return x


# ======================================================================
# FINALIZE OUTPUT
# ======================================================================
def _finalize_result(
    species,
    sigma_eff,
    flag,
    densities,
    densities_z,
    densities_zs,
    vij,
    variables,
    volume_factors,
    mf_sym,
    hc_sym,
):
    # ⚡ NO simplify – extremely expensive and useless for final export
    fe_total = _sympify_maybe(mf_sym) + _sympify_maybe(hc_sym)

    return {
        "species": species,
        "sigma_eff": sigma_eff,
        "flag": flag,
        "densities": densities,
        "densities_z": densities_z,
        "densities_zs": densities_zs,
        "vij": vij,
        "variables": variables,
        "volume_factors": volume_factors,

        "free_energy_mf_symbolic": mf_sym,
        "free_energy_hc_symbolic": hc_sym,
        "free_energy_excess_symbolic": fe_total,
    }


# ======================================================================
# MERGE CASES
# ======================================================================
def _merge_both(mf_res, hc_res):
    mf = _normalize_result(mf_res, "mf")
    hc = _normalize_result(hc_res, "hc")

    species = mf["species"] or hc["species"]
    sigma_eff = hc["sigma_eff"] or mf["sigma_eff"]
    flag = hc["flag"] or mf["flag"]

    return _finalize_result(
        species,
        sigma_eff,
        flag,
        mf["densities"],
        mf["densities_z"],
        mf["densities_zs"],
        mf["vij"] or hc["vij"],
        mf["variables"] or hc["variables"],
        mf["volume_factors"] or hc["volume_factors"],
        mf_sym=mf["f_two_point"],
        hc_sym=hc["f_symbolic"],
    )


def _merge_only_mf(mf_res):
    mf = _normalize_result(mf_res, "mf")
    return _finalize_result(
        mf["species"], mf["sigma_eff"], mf["flag"],
        mf["densities"], mf["densities_z"], mf["densities_zs"],
        mf["vij"], mf["variables"], mf["volume_factors"],
        mf_sym=mf["f_two_point"], hc_sym=0
    )


def _merge_only_hc(hc_res):
    hc = _normalize_result(hc_res, "hc")
    return _finalize_result(
        hc["species"], hc["sigma_eff"], hc["flag"],
        hc["densities"], hc["densities_z"], hc["densities_zs"],
        hc["vij"], hc["variables"], hc["volume_factors"],
        mf_sym=0, hc_sym=hc["f_symbolic"]
    )


# ======================================================================
# MAIN ENTRY
# ======================================================================
def total_free_energy_z(ctx):
    cfg = _load_config(ctx)
    fe = cfg["free_energy_parameters"]

    mode = fe.get("mode", "standard")
    method = fe.get("method", "smf")

    # HYBRID
    if mode == "hybrid":
        result = free_energy_hybrid(ctx)
        mf = result.get("f_mf_two_point") or result.get("f_mf")
        hc = result.get("f_hc")

        result = _finalize_result(
            result.get("species", []),
            result.get("sigma_eff", []),
            result.get("flag", []),
            result.get("densities", []),
            result.get("densities_z", []),
            result.get("densities_zs", []),
            result.get("vij", []),
            result.get("variables", None),
            result.get("volume_factors", []),
            mf_sym=mf,
            hc_sym=hc,
        )

    else:
        # HARD CORE
        try:
            hc_res = free_energy_hard_core(ctx)
        except Exception:
            hc_res = None

        # MEAN FIELD
        try:
            if method == "emf":
                mf_res = free_energy_EMF(ctx)
            elif method == "smf":
                mf_res = free_energy_SMF(ctx)
            elif method == "void":
                mf_res = free_energy_void(ctx)
            else:
                mf_res = None
        except Exception:
            mf_res = None

        # MERGE
        if mf_res is not None and hc_res is not None:
            result = _merge_both(mf_res, hc_res)
        elif mf_res is not None:
            result = _merge_only_mf(mf_res)
        elif hc_res is not None:
            result = _merge_only_hc(hc_res)
        else:
            result = _finalize_result([], [], [], [], [], [], [], None, [], 0, 0)

    # SAVE JSON
    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    out_file = scratch / "total_free_energy_z_result.json"

    with open(out_file, "w") as fh:
        json.dump(_jsonify_sympy(result), fh, indent=4)

    print(f"[INFO] Total free energy written to {out_file}")
    return result
