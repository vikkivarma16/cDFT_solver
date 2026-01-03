from pathlib import Path
from collections.abc import Mapping

from .void import free_energy_void
from .EMF import free_energy_EMF
from .SMF import free_energy_SMF


def mean_field(
    ctx=None,
    hc_data=None,
    system_config=None,
    export_json=True,
    filename=None,
):
    """
    Unified wrapper for symbolic free energy models (EMF, SMF, VOID).
    """

    if system_config is None:
        raise ValueError("system_config must be provided")

    # ============================================================
    # Recursive key lookup (robust)
    # ============================================================
    def find_key_recursive(obj, key):
        """
        Recursively find FIRST occurrence of key in nested structures.
        """
        if isinstance(obj, Mapping):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = find_key_recursive(v, key)
                if found is not None:
                    return found

        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found = find_key_recursive(item, key)
                if found is not None:
                    return found

        return None

    # ------------------------------------------------------------
    # Extract system + method
    # ------------------------------------------------------------
    system = find_key_recursive(system_config, "system")
    method = find_key_recursive(system_config, "method")

    if system is None:
        raise ValueError("Could not locate 'system' section in system_config")

    if method is None:
        raise ValueError("Could not locate 'method' in system_config")

    method = str(method).lower()

    # ------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------
    dispatch = {
        "emf": {
            "func": free_energy_EMF,
            "default_filename": "Solution_EMF.json",
        },
        "smf": {
            "func": free_energy_SMF,
            "default_filename": "Solution_SMF.json",
        },
        "void": {
            "func": free_energy_void,
            "default_filename": "Solution_void.json",
        },
    }

    if method not in dispatch:
        raise ValueError(
            f"Unknown free energy method '{method}'. "
            f"Available methods: {list(dispatch.keys())}"
        )

    entry = dispatch[method]
    func = entry["func"]
    out_filename = filename or entry["default_filename"]

    # ------------------------------------------------------------
    # Call selected model
    # ------------------------------------------------------------
    result = func(
        ctx=ctx,
        hc_data=hc_data,
        export_json=export_json,
        filename=out_filename,
    )

    # ------------------------------------------------------------
    # Attach metadata
    # ------------------------------------------------------------
    result["system"] = system
    result["method"] = method

    return result
