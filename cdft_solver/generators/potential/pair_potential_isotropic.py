
from .pair_potential_isotropic_registry import get_isotropic_pair_potential_factory # noqa

from .pair_potential_isotropic_default import pair_potential_isotropic_default  # noqa

def pair_potential_isotropic(specific_pair_potential):
    """
    Vectorized isotropic pair potential dispatcher.
    """
    factory = get_isotropic_pair_potential_factory(specific_pair_potential)
    return factory(specific_pair_potential)



'''
updating the registry 

from cdft_solver.generators.potential.pair_potential_isotropic_registry import (
    register_isotropic_pair_potential
)
import numpy as np

def yukawa_factory(p):
    epsilon = p.get("epsilon", 1.0)
    kappa = p.get("kappa", 1.0)

    def V(r):
        r = np.asarray(r)
        return epsilon * np.exp(-kappa * r) / r

    return V

register_isotropic_pair_potential("yukawa", yukawa_factory)




'''
