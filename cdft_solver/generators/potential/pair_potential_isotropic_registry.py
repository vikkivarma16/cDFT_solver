import numpy as np
from copy import deepcopy

# ============================================================
# GLOBAL REGISTRY
# ============================================================

ISOTROPIC_PAIR_POTENTIAL_REGISTRY = {}
# pair_potential_isotropic.py

# IMPORTANT: import for side effects
from . import pair_potential_isotropic_default  # noqa



def pair_potential_isotropic(specific_pair_potential):
    factory = get_isotropic_pair_potential_factory(specific_pair_potential)
    return factory(specific_pair_potential)


def register_isotropic_pair_potential(
    potential_type,
    factory_fn,
    overwrite=False,
):
    """
    Register an isotropic pair potential factory.

    Parameters
    ----------
    potential_type : str
        Name used in input dict: potential["type"]
    factory_fn : callable
        Function: dict -> callable(r)
    overwrite : bool
        Allow overwriting existing registration
    """
    key = potential_type.lower()

    if key in ISOTROPIC_PAIR_POTENTIAL_REGISTRY and not overwrite:
        raise KeyError(
            f"Isotropic potential '{key}' already registered"
        )

    ISOTROPIC_PAIR_POTENTIAL_REGISTRY[key] = factory_fn


def get_isotropic_pair_potential_factory(potential):
    """
    Resolve factory from registry.
    """
    if not isinstance(potential, dict):
        raise TypeError("Potential specification must be a dict")

    pt = potential.get("type", "").lower()
    if pt not in ISOTROPIC_PAIR_POTENTIAL_REGISTRY:
        raise ValueError(f"Unknown potential type: {pt}")

    return ISOTROPIC_PAIR_POTENTIAL_REGISTRY[pt]
    
    
    


