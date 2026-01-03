# cdft_solver/registries/potential_converters.py

from copy import deepcopy

# ---------------------------------------------------------
# Global registry for potential conversion
# ---------------------------------------------------------
POTENTIAL_CONVERTER_REGISTRY = {}


def register_potential_converter(potential_type, converter_fn, overwrite=False):
    """
    Register a potential conversion function.

    Parameters
    ----------
    potential_type : str
        Input potential type (lowercase)
    converter_fn : callable
        Function: dict -> dict
    overwrite : bool
        If True, overwrite existing converter
    """
    key = potential_type.lower()

    if key in POTENTIAL_CONVERTER_REGISTRY and not overwrite:
        raise KeyError(
            f"Converter for potential type '{key}' already registered"
        )

    POTENTIAL_CONVERTER_REGISTRY[key] = converter_fn


def convert_potential_via_registry(potential):
    """
    Convert a potential dictionary using the registry.
    Falls back to identity conversion if no rule exists.
    """
    if not isinstance(potential, dict):
        raise TypeError("Potential must be a dictionary")

    ptype = potential.get("type", "").lower()
    converter = POTENTIAL_CONVERTER_REGISTRY.get(ptype)

    if converter is None:
        return deepcopy(potential)

    return converter(deepcopy(potential))
    
    
    
# cdft_solver/registries/potential_converters.py (continued)

# -----------------------
# Default converters
# -----------------------

def _hc_to_zero(pot):
    return {"type": "zero_potential"}

def _lj_to_salj(pot):
    pot["type"] = "salj"
    return pot

def _mie_to_ma(pot):
    pot["type"] = "ma"
    return pot


register_potential_converter("hc", _hc_to_zero)
register_potential_converter("ghc", _hc_to_zero)
register_potential_converter("lj", _lj_to_salj)
register_potential_converter("mie", _mie_to_ma)



''' 
updatation inline

from cdft_solver.generators.potential_splitter.mf_potential_registry import (
    register_potential_converter
)

def yukawa_to_mf(pot):
    pot["type"] = "screened_yukawa_mf"
    return pot

register_potential_converter("yukawa", yukawa_to_mf)
'''





