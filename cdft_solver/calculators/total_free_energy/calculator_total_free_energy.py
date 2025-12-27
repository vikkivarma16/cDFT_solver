import json
from pathlib import Path
import sympy as sp
import ast

from cdft_solver.calculators.free_energy_hard_core.calculator_free_energy_hard_core import free_energy_hard_core
from cdft_solver.calculators.free_energy_mean_field.calculator_free_energy_EMF import free_energy_EMF
from cdft_solver.calculators.free_energy_mean_field.calculator_free_energy_SMF import free_energy_SMF
from cdft_solver.calculators.free_energy_mean_field.calculator_free_energy_void import free_energy_void
from cdft_solver.calculators.free_energy_hybrid.calculator_free_energy_hybrid import free_energy_hybrid
from cdft_solver.calculators.free_energy_ideal.calculator_free_energy_ideal import free_energy_ideal


# -------------------------
# CONFIG LOADING
# -------------------------
def _load_config(ctx):
    scratch = Path(ctx.scratch_dir)
    cfg_path_scratch = scratch / "input_data_free_energy_parameters.json"
    input_file = getattr(ctx, "input_file", None)
    cfg_path_input = Path(input_file) if input_file else None

    if cfg_path_scratch.exists():
        path = cfg_path_scratch
    elif cfg_path_input and cfg_path_input.exists():
        path = cfg_path_input
    else:
        return {"free_energy_parameters": {"mode": "standard", "method": "smf"}}

    with open(path, "r") as fh:
        cfg = json.load(fh)

    fe = cfg.get("free_energy") or cfg.get("free_energy_parameters") or {}
    return {
        "free_energy_parameters": {
            "mode": fe.get("mode", "standard").lower(),
            "method": fe.get("method", "smf").lower(),
            "integrated_strength_kernel": fe.get("integrated_strength_kernel", None),
            "supplied_data": fe.get("supplied_data", None),
        }
    }


# -------------------------
# NORMALIZATION
# -------------------------
def _normalize_result(res, kind):
    norm = {}

    def _parse_list(val):
        if val is None:
            return []
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            except Exception:
                return [val]
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]

    if res is None:
        norm["species"] = []
        norm["sigma_eff"] = []
        norm["flag"] = []
        norm["densities"] = []
        norm["vij"] = []
        norm["volume_factors"] = []
        norm["f_symbolic"] = None
        return norm

    norm["species"] = _parse_list(res.get("species", []))
    norm["sigma_eff"] = _parse_list(res.get("sigma_eff", []))
    norm["flag"] = _parse_list(res.get("flag", []))
    norm["densities"] = _parse_list(res.get("densities", []))
    norm["vij"] = _parse_list(res.get("vij", []))
    norm["volume_factors"] = _parse_list(res.get("volume_factors", []))

    if kind == "hc":
        norm["f_symbolic"] = res.get("f_hc")
    elif kind == "mf":
        norm["f_symbolic"] = res.get("f_mf")
    elif kind == "ideal":
        norm["f_symbolic"] = res.get("f_ideal")
    else:
        norm["f_symbolic"] = None

    return norm


# -------------------------
# SYMPIFY HELPER
# -------------------------
def _sympify_maybe(expr):
    if expr is None:
        return 0
    if isinstance(expr, sp.Expr):
        return expr
    if isinstance(expr, str):
        try:
            return sp.sympify(expr)
        except Exception:
            return 0
    try:
        return sp.sympify(str(expr))
    except Exception:
        return 0


# -------------------------
# FINALIZE RESULT
# -------------------------
def _finalize_result(species, sigma_eff, flag, densities, vij, volume_factors, mf_res, hc_res, ideal_res, f_total):
    result = {
        "species": species,
        "sigma_eff": sigma_eff,
        "flag": flag,
        "densities": densities,
        "vij": vij,
        "volume_factors": volume_factors,
        "mf_module_raw": mf_res,
        "hc_module_raw": hc_res,
        "ideal_module_raw": ideal_res,
        "free_energy_symbolic": f_total,
        "free_energy_total_numeric": None,
    }

    if f_total is not None:
        try:
            result["free_energy_total_numeric"] = float(f_total.evalf())
        except Exception:
            result["free_energy_total_numeric"] = None

    return result


# -------------------------
# MERGE FUNCTIONS
# -------------------------
def _merge_both(mf_res, hc_res, ideal_res):
    hc = _normalize_result(hc_res, "hc")
    mf = _normalize_result(mf_res, "mf")
    ideal = _normalize_result(ideal_res, "ideal")

    species = mf["species"] or hc["species"] or ideal["species"]
    sigma_eff = hc["sigma_eff"] or mf["sigma_eff"] or ideal["sigma_eff"]
    flag = hc["flag"] or mf["flag"] or ideal["flag"]
    densities = mf["densities"] or hc["densities"] or ideal["densities"]
    vij = mf["vij"] or hc["vij"] or ideal["vij"]
    volume_factors = mf["volume_factors"] or hc["volume_factors"] or ideal["volume_factors"]

    sf_mf = _sympify_maybe(mf.get("f_symbolic"))
    sf_hc = _sympify_maybe(hc.get("f_symbolic"))
    sf_id = _sympify_maybe(ideal.get("f_symbolic"))
    total_symbolic = sp.simplify(sf_mf + sf_hc + sf_id)

    return _finalize_result(species, sigma_eff, flag, densities, vij, volume_factors, mf_res, hc_res, ideal_res, total_symbolic)


def _merge_only_mf(mf_res, ideal_res):
    mf = _normalize_result(mf_res, "mf")
    ideal = _normalize_result(ideal_res, "ideal")

    species = mf["species"] or ideal["species"]
    sigma_eff = mf["sigma_eff"] or ideal["sigma_eff"]
    flag = mf["flag"] or ideal["flag"]
    densities = mf["densities"] or ideal["densities"]
    vij = mf["vij"] or ideal["vij"]
    volume_factors = mf["volume_factors"] or ideal["volume_factors"]

    sf_mf = _sympify_maybe(mf.get("f_symbolic"))
    sf_id = _sympify_maybe(ideal.get("f_symbolic"))
    total_symbolic = sp.simplify(sf_mf + sf_id)

    return _finalize_result(species, sigma_eff, flag, densities, vij, volume_factors, mf_res, None, ideal_res, total_symbolic)


def _merge_only_hc(hc_res, ideal_res):
    hc = _normalize_result(hc_res, "hc")
    ideal = _normalize_result(ideal_res, "ideal")

    species = hc["species"] or ideal["species"]
    sigma_eff = hc["sigma_eff"] or ideal["sigma_eff"]
    flag = hc["flag"] or ideal["flag"]
    densities = hc["densities"] or ideal["densities"]
    vij = hc["vij"] or ideal["vij"]
    volume_factors = hc["volume_factors"] or ideal["volume_factors"]

    sf_hc = _sympify_maybe(hc.get("f_symbolic"))
    sf_id = _sympify_maybe(ideal.get("f_symbolic"))
    total_symbolic = sp.simplify(sf_hc + sf_id)

    return _finalize_result(species, sigma_eff, flag, densities, vij, volume_factors, None, hc_res, ideal_res, total_symbolic)


def _merge_results(mf_res, hc_res, ideal_res):
    if mf_res is None and hc_res is None:
        ideal = _normalize_result(ideal_res, "ideal")
        f_id = _sympify_maybe(ideal.get("f_symbolic"))
        return _finalize_result(
            ideal["species"], ideal["sigma_eff"], ideal["flag"], ideal["densities"],
            ideal["vij"], ideal["volume_factors"], None, None, ideal_res, f_id
        )

    if mf_res is not None and hc_res is not None:
        return _merge_both(mf_res, hc_res, ideal_res)
    if mf_res is not None and hc_res is None:
        return _merge_only_mf(mf_res, ideal_res)
    if hc_res is not None and mf_res is None:
        return _merge_only_hc(hc_res, ideal_res)


# -------------------------
# RECURSIVE JSONIFY
# -------------------------
def _jsonify_sympy(obj):
    if isinstance(obj, (sp.Basic, sp.Expr)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _jsonify_sympy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_jsonify_sympy(v) for v in obj]
    else:
        return obj


# -------------------------
# TOTAL FREE ENERGY
# -------------------------
def total_free_energy(ctx):
    cfg = _load_config(ctx)
    fe_cfg = cfg.get("free_energy_parameters", {})
    mode = fe_cfg.get("mode", "standard").lower()
    method = fe_cfg.get("method", "smf").lower()

    allowed_methods = ("emf", "smf", "void")

    ideal_res = free_energy_ideal(ctx)

    if mode == "hybrid":
        try:
            result = free_energy_hybrid(ctx)
        except Exception as e:
            raise RuntimeError(f"Hybrid free energy calculation failed: {e}")

    elif mode == "standard":
        if method not in allowed_methods:
            raise ValueError(f"Unsupported method '{method}'. Allowed: {allowed_methods}")

        try:
            hc_res = free_energy_hard_core(ctx)
        except Exception as e:
            hc_res = None
            print(f"[WARN] Hard-core free energy contribution is zero: {e}")

        try:
            if method == "emf":
                mf_res = free_energy_EMF(ctx)
            elif method == "smf":
                mf_res = free_energy_SMF(ctx)
            elif method == "void":
                mf_res = free_energy_void(ctx)
            else:
                mf_res = None
        except Exception as e:
            mf_res = None
            print(f"[WARN] Mean-field free energy contribution is zero: {e}")

        result = _merge_results(mf_res, hc_res, ideal_res)

    else:
        raise ValueError(f"Unsupported free-energy mode '{mode}'. Use 'standard' or 'hybrid'.")

    scratch = Path(ctx.scratch_dir)
    scratch.mkdir(exist_ok=True)
    out_path = scratch / "total_free_energy_result.json"

    json_ready = _jsonify_sympy(result)
    with open(out_path, "w") as fh:
        json.dump(json_ready, fh, indent=4)

    print(f"[INFO] Total free energy results written to {out_path}")
    return result

