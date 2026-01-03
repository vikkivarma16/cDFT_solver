from pathlib import Path
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

    if system_config is None or "system" not in system_config:
        raise ValueError("system_config must contain a 'system' section")

    
    def find_key_recursive(obj, key):
        """
        Recursively find a key in nested mappings (dict, OrderedDict, etc).
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
    
    method = find_key_recursive(system_config,"method")
    print(method)

    # -------------------------
    # Dispatch table
    # -------------------------
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

