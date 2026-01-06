from pathlib import Path

from .void_z import free_energy_void_z
from .EMF_z import free_energy_EMF_z
from .SMF_z import free_energy_SMF_z

def mean_field_planer(
    ctx=None,
    hc_data=None,
    system_config=None,
    export_json=True,
    filename=None,
):
    """
    Unified wrapper for symbolic free energy models (EMF, SMF, VOID).

    Parameters
    ----------
    ctx : object, optional
        Context object with `scratch_dir`
    hc_data : dict
        Species / hard-core data (passed directly to model functions)
    system_config : dict
        Control dictionary, e.g.
        {
            "system": {
                "mode": "standard",
                "method": "emf",
                "integrated_strength_kernel": "rdf",
                "supplied_data": "no"
            }
        }
    export_json : bool
        Whether to export symbolic data to JSON
    filename : str, optional
        Override default output filename

    Returns
    -------
    dict
        Output of the selected free energy function
    """

    
    def find_key_recursive(d, key):
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

    method = find_key_recursive(system_config, "method").lower()

    # -------------------------
    # Dispatch table
    # -------------------------
    dispatch = {
        "emf": {
            "func": free_energy_EMF_z,
            "default_filename": "Solution_EMF.json",
        },
        "smf": {
            "func": free_energy_SMF_z,
            "default_filename": "Solution_SMF.json",
        },
        "void": {
            "func": free_energy_void_z,
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

    # -------------------------
    # Call selected model
    # -------------------------
    result = func(
        ctx=ctx,
        hc_data=hc_data,
        export_json=export_json,
        filename=out_filename,
    )

    # -------------------------
    # Attach system metadata
    # -------------------------
    result["system"] = system
    result["method"] = method

    return result

