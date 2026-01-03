# ============================================================
# TOTAL FREE ENERGY JSON EXPORTER
# ============================================================

import json
from pathlib import Path
import sympy as sp


def _serialize_symbol(sym):
    """Serialize a SymPy symbol safely."""
    return {
        "name": sym.name,
        "assumptions": {
            k: bool(v) for k, v in sym.assumptions0.items()
        }
    }


def _serialize_expression(expr):
    """Serialize a SymPy expression (lossless)."""
    return sp.srepr(expr)


def free_energy_exporter(
    ctx,
    total_fe,
    filename,
    system_config=None,
    indent=2,
):
    """
    Export merged total free energy to a JSON file.

    Parameters
    ----------
    total_fe : dict
        Output of total_free_energy()
    filename : str or Path
        Output JSON file
    system_config : dict, optional
        System metadata
    """

    data = {
        "selected_model": total_fe.get("selected_model"),
        "variables": [
            _serialize_symbol(v) for v in total_fe["variables"]
        ],
        "lambda_arguments": [v.name for v in total_fe["variables"]],
        "expression": _serialize_expression(total_fe["expression"]),
        "components": [],
    }

    if system_config is not None:
        data["system"] = system_config.get("system", system_config)

    # ------------------------------------------------------------
    # Component-wise export
    # ------------------------------------------------------------
    for comp in total_fe["components"]:
        data["components"].append({
            "variables": [v.name for v in comp["variables"]],
            "expression": _serialize_expression(comp["expression"]), })
            
    
    scratch = Path(ctx.scratch_dir)

    filename = Path(scratch/filename)
    with filename.open("w") as f:
        json.dump(data, f, indent=indent)
        
    print(f"✅ Total free energy exported to: {filename}")

    return filename

