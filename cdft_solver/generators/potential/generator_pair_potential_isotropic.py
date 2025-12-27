import numpy as np

def pair_potential_isotropic(specific_pair_potential):
    """
    Vectorized isotropic pair potential based on a dictionary input.
    Works for arrays of r.
    """
    pt = specific_pair_potential["type"].lower()
    sigma = specific_pair_potential.get("sigma", 1.0)
    epsilon = specific_pair_potential.get("epsilon", 1.0)
    cutoff = specific_pair_potential.get("cutoff", 5.0)
    EPS = 1e-4
   

    if pt in ["zero", "zero_potential"]:
        def V(r):
            r = np.asarray(r)
            return np.zeros_like(r)
        return V

    elif pt == "custom_1" :
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r < r_cutoff] = -epsilon
            mask = (r >= r_cutoff) & (r < cutoff)
            v[mask] = 4 * epsilon * ((sigma / r[mask]) ** 12 - 2 * (sigma / r[mask]) ** 6)
            v[r >= cutoff] = 0.0
            return v
        return V

    elif pt == "custom_3":
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r <= EPS] = 2e9
            mask = (r > EPS) & (r < cutoff)
            v[mask] = epsilon * ((2.0 / 15.0) * (sigma / r[mask]) ** 9 - (sigma / r[mask]) ** 3)
            v[r >= cutoff] = 0.0
            
            #print ("hey I am being accessed")
            return v
        return V

    elif pt in ["hard_core", "hard_sphere", "hc", "ghc"]:
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r <= sigma] = 2e16
            v[r > sigma] = 0.0
            return v
        return V

    elif pt == "mie":
        n = specific_pair_potential.get("n", 12)
        m = specific_pair_potential.get("m", 6)
        c_mie = (n / (n - m)) * (n / m) ** (m / (n - m))
        r_cutoff = 2 ** (1 / n-m) * sigma
        def V(r):
            r = np.asarray(r)
            v = epsilon * c_mie * ((sigma / r) ** n - (sigma / r) ** m)
            v[r > cutoff] = 0.0
            return v
        return V

    elif pt in ["gaussian", "gs"]:
        def V(r):
            r = np.asarray(r)
            return epsilon * np.exp(-(r / sigma) ** 2)
        return V

    elif pt in ["lennard-jones", "lj"] :
        def V(r):
            r = np.asarray(r)
            return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        return V

    elif pt == "wca":
        r_cutoff = 2 ** (1 / 6) * sigma
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r < r_cutoff] = -epsilon
            mask = (r >= r_cutoff) & (r < cutoff)
            v[mask] = 4 * epsilon * ((sigma / r[mask]) ** 12 - 2 * (sigma / r[mask]) ** 6)
            return v
        return V

    elif pt == "ma":
        m = specific_pair_potential.get("m", 12.0)
        n = specific_pair_potential.get("n", 6.0)
        lambda_p = specific_pair_potential.get("lambda", 1.0)
        r_cutoff = 2 ** (1 / m-n) * sigma
        def V(r):
            r = np.asarray(r)
            r_min = (lambda_p * m / n) ** (1 / (m - n))
            cutoff_val = cutoff
            v = np.zeros_like(r)
            v[r < r_min] = -epsilon / lambda_p
            mask = (r >= r_min)
            v[mask] = (m / (m - n)) * ((m / n) ** (n / (m - n))) * epsilon * (
                lambda_p * (sigma / r[mask]) ** m - (sigma / r[mask]) ** n
            )
        
            return v
        return V

    else:
        raise ValueError(f"Unknown potential type: {pt}")


def pair_potential_isotropic_integrant(r, specific_pair_potential):
    """
    Scalar potential for a single distance r using the same dictionary input style.
    Works for integration.
    """
    pt = specific_pair_potential["type"].lower()
    sigma = specific_pair_potential.get("sigma", 1.0)
    epsilon = specific_pair_potential.get("epsilon", 1.0)
    cutoff = specific_pair_potential.get("cutoff", 5.0)
    EPS = 1e-4
    r_cutoff = 2 ** (1 / 6) * sigma

    if pt in ["zero", "zero_potential"]:
        return 0.0

    elif pt == "custom_1" or pt == "custom_2":
        if r < r_cutoff:
            return -epsilon
        elif r_cutoff <= r < cutoff:
            return 4 * epsilon * ((sigma / r) ** 12 - 2 * (sigma / r) ** 6)
        else:
            return 0.0

    elif pt == "custom_3":
        if r <= EPS:
            return 2e7
        elif r < cutoff:
            return epsilon * ((2.0 / 15.0) * (sigma / r) ** 9 - (sigma / r) ** 3)
        else:
            return 0.0

    elif pt in ["hard_core", "hard_sphere", "hc"]:
        return 2e16 if r <= sigma else 0.0

    elif pt == "mie":
        n = specific_pair_potential.get("n", 12)
        m = specific_pair_potential.get("m", 6)
        if r <= EPS:
            return 2e8
        elif r < cutoff:
            c_mie = (n / (n - m)) * (n / m) ** (m / (n - m))
            return epsilon * c_mie * ((sigma / r) ** n - (sigma / r) ** m)
        else:
            return 0.0

    elif pt == "gaussian":
        return epsilon * np.exp(-(r / sigma) ** 2)

    elif pt == "lennard-jones":
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    elif pt == "wca":
        if r < r_cutoff:
            return -epsilon
        elif r_cutoff <= r < cutoff:
            return 4 * epsilon * ((sigma / r) ** 12 - 2 * (sigma / r) ** 6)
        else:
            return 0.0

    elif pt == "ma":
        m = specific_pair_potential.get("m", 12.0)
        n = specific_pair_potential.get("n", 6.0)
        lambda_p = specific_pair_potential.get("lambda", 1.0)
        r_min = (lambda_p * m / n) ** (1 / (m - n))
        if r < r_min:
            return -epsilon / lambda_p
        elif r_min <= r < cutoff:
            return (m / (m - n)) * ((m / n) ** (n / (m - n))) * epsilon * (
                lambda_p * (sigma / r) ** m - (sigma / r) ** n
            )
        else:
            return 0.0

    else:
        raise ValueError(f"Unknown potential type: {pt}")


# ===============================================================
# Example usage
# ===============================================================
if __name__ == "__main__":
    potentials = [
        {"type": "zero", "sigma": 1.0, "epsilon": 1.0, "cutoff": 5.0},
        {"type": "custom_1", "sigma": 1.0, "epsilon": 1.0, "cutoff": 5.0},
        {"type": "custom_3", "sigma": 1.0, "epsilon": 1.0, "cutoff": 5.0},
        {"type": "hard_core", "sigma": 1.0, "epsilon": 1.0},
        {"type": "mie", "sigma": 1.0, "epsilon": 1.0, "n": 12, "m": 6, "cutoff": 5.0},
        {"type": "gaussian", "sigma": 1.0, "epsilon": 1.0},
        {"type": "ma", "sigma": 1.0, "epsilon": 0.1787, "m": 12, "n": 6, "lambda": 0.477246, "cutoff": 3.5}
    ]

    r = np.linspace(0.001, 6.0, 100)

    for p in potentials:
        V_vec = pair_potential_isotropic(p)
        v_r_vec = V_vec(r)
        v_scalar = pair_potential_isotropic_integrant(1.0, p)
        print(f"Potential type: {p['type']}")
        print("Vectorized first 10:", v_r_vec[:10])
        print("Scalar at r=1.0:", v_scalar)
        print()

