import numpy as np
from cdft_solver.generators.potential.pair_potential_isotropic_registry import (
    register_isotropic_pair_potential
)

EPS = 1e-4

def pair_potential_isotropic_default():
    # ------------------------------------------------------------
    # ZERO
    # ------------------------------------------------------------
    def zero_potential(p):
        def V(r):
            return np.zeros_like(np.asarray(r))
        return V

    register_isotropic_pair_potential("zero", zero_potential)
    register_isotropic_pair_potential("zero_potential", zero_potential)


    # ------------------------------------------------------------
    # HARD CORE
    # ------------------------------------------------------------
    def hard_core(p):
        sigma = p.get("sigma", 1.0)
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r <= sigma] = 2e16
            return v
        return V

    for name in ["hard_core", "hard_sphere", "hc", "ghc"]:
        register_isotropic_pair_potential(name, hard_core)


    # ------------------------------------------------------------
    # LENNARD-JONES
    # ------------------------------------------------------------
    def lj(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        def V(r):
            r = np.asarray(r)
            return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        return V

    register_isotropic_pair_potentials = register_isotropic_pair_potential
    register_isotropic_pair_potentials("lj", lj)
    register_isotropic_pair_potentials("lennard-jones", lj)


    # ------------------------------------------------------------
    # GAUSSIAN
    # ------------------------------------------------------------
    def gaussian(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        def V(r):
            r = np.asarray(r)
            return epsilon * np.exp(-(r / sigma) ** 2)
        return V

    register_isotropic_pair_potential("gaussian", gaussian)
    register_isotropic_pair_potential("gs", gaussian)


    # ------------------------------------------------------------
    # MIE
    # ------------------------------------------------------------
    def mie(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        n = p.get("n", 12)
        m = p.get("m", 6)
        cutoff = p.get("cutoff", 5.0)

        c = (n / (n - m)) * (n / m) ** (m / (n - m))

        def V(r):
            r = np.asarray(r)
            v = epsilon * c * ((sigma / r) ** n - (sigma / r) ** m)
            v[r > cutoff] = 0.0
            return v

        return V

    register_isotropic_pair_potential("mie", mie)



    EPS = 1e-4




    # ============================================================
    # CUSTOM_1
    # ============================================================

    def custom_1(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        cutoff = p.get("cutoff", 5.0)

        # NOTE: original code implicitly used r_cutoff
        r_cutoff = p.get("r_cutoff", 2 ** (1 / 6) * sigma)

        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)

            v[r < r_cutoff] = -epsilon

            mask = (r >= r_cutoff) & (r < cutoff)
            v[mask] = 4 * epsilon * (
                (sigma / r[mask]) ** 12 - 2 * (sigma / r[mask]) ** 6
            )

            v[r >= cutoff] = 0.0
            return v

        return V


    register_isotropic_pair_potential("custom_1", custom_1)


    # ============================================================
    # CUSTOM_3
    # ============================================================

    def custom_3(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        cutoff = p.get("cutoff", 5.0)

        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)

            v[r <= EPS] = 2e9

            mask = (r > EPS) & (r < cutoff)
            v[mask] = epsilon * (
                (2.0 / 15.0) * (sigma / r[mask]) ** 9
                - (sigma / r[mask]) ** 3
            )

            v[r >= cutoff] = 0.0
            return v

        return V


    register_isotropic_pair_potential("custom_3", custom_3)





    # ============================================================
    # WCA
    # ============================================================

    def wca(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        cutoff = p.get("cutoff", 5.0)

        r_cutoff = 2 ** (1 / 6) * sigma

        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)

            v[r < r_cutoff] = -epsilon

            mask = (r >= r_cutoff) & (r < cutoff)
            v[mask] = 4 * epsilon * (
                (sigma / r[mask]) ** 12 - 2 * (sigma / r[mask]) ** 6
            )

            return v

        return V


    register_isotropic_pair_potential("wca", wca)



    # ============================================================
    # MA
    # ============================================================

    def ma(p):
        sigma = p.get("sigma", 1.0)
        epsilon = p.get("epsilon", 1.0)
        cutoff = p.get("cutoff", 5.0)

        m = p.get("m", 12.0)
        n = p.get("n", 6.0)
        lambda_p = p.get("lambda", 1.0)

        def V(r):
            r = np.asarray(r)

            r_min = (lambda_p * m / n) ** (1 / (m - n))

            v = np.zeros_like(r)

            v[r < r_min] = -epsilon / lambda_p

            mask = (r >= r_min)
            v[mask] = (
                (m / (m - n))
                * ((m / n) ** (n / (m - n)))
                * epsilon
                * (
                    lambda_p * (sigma / r[mask]) ** m
                    - (sigma / r[mask]) ** n
                )
            )

            return v

        return V


    register_isotropic_pair_potential("ma", ma)

