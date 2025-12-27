import numpy as np

"""
Unified Pair Potential Module (1D radial)
----------------------------------------

This version implements EXACT the potentials you defined in
interaction_potential_r_1d(), but now using the same RETURN STYLE
as pair_potential_isotropic().

Meaning:

    V = pair_potential_one_d(specific_pair_potential)
    v_r = V(r_array)

And also:

    v_scalar = pair_potential_one_d_integrant(r, specific_pair_potential)

Supported potential types:
    - "wca"
    - "mie"
    - "ma"
    - "gs"
    - "yk"
    - "hc"
    - "zero"
"""


# =====================================================================
#  VECTORIZED FUNCTION-GENERATOR  (returns V(r) as a callable)
# =====================================================================

def pair_potential_one_d(specific_pair_potential):
    """
    Returns a VECTORIZED function V(r) following the return style of
    pair_potential_isotropic().

    Input dictionary example:
        {
            "type": "wca",
            "epsilon": 1.0,
            "sigma": 1.0,
            "cutoff": 5.0
        }

    Usage:
        V = pair_potential_one_d(pot)
        values = V(r_array)
    """

    pt = specific_pair_potential["type"].lower()
    sigma = specific_pair_potential.get("sigma", 1.0)
    epsilon = specific_pair_potential.get("epsilon", 1.0)
    cutoff = specific_pair_potential.get("cutoff", 5.0)
    EPS = 1e-4

    # ==============================================================
    # ZERO
    # ==============================================================
    if pt in ["zero", "zero_potential"]:
        def V(r):
            r = np.asarray(r)
            return np.zeros_like(r)
        return V

    # ==============================================================
    # WCA
    # ==============================================================
    elif pt == "wca":
        r_cutoff = 2**(1/6) * sigma
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r < r_cutoff] = -epsilon
            mask = (r >= r_cutoff) & (r < cutoff)
            v[mask] = 4*epsilon*((sigma/r[mask])**12 - 2*(sigma/r[mask])**6)
            return v
        return V

    # ==============================================================
    # MIE  (your definition uses 48-24)
    # ==============================================================
    elif pt == "mie":
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            mask = (r >= sigma) & (r < cutoff)
            v[mask] = -4*epsilon*((sigma/r[mask])**48 - (sigma/r[mask])**24)
            return v
        return V

    # ==============================================================
    # MA (multi-Yukawa–shape)
    # ==============================================================
    elif pt == "ma":
        m = float(specific_pair_potential.get("m", 12))
        n = float(specific_pair_potential.get("n", 6))
        lambda_p = float(specific_pair_potential.get("lambda", 1.0))
        r_min = sigma*(lambda_p*m/n)**(1/(m-n))
        factor = m/(m-n)*((m/n)**(n/(m-n)))
        cutoff_val = cutoff

        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)

            mask1 = (np.abs(r) < r_min)
            mask2 = (np.abs(r) >= r_min) & (np.abs(r) < cutoff_val)

            # inner parabola-like
            v[mask1] = -np.pi*epsilon/lambda_p*(r_min**2 - r[mask1]**2) + \
                       factor*2*np.pi*epsilon*sigma**2 * (
                        (lambda_p/(2-m)*(sigma/cutoff_val)**(m-2)
                         - 1/(2-n)*(sigma/cutoff_val)**(n-2))
                        -
                        (lambda_p/(2-m)*(sigma/r_min)**(m-2)
                         - 1/(2-n)*(sigma/r_min)**(n-2))
                       )

            # tail region
            v[mask2] = factor*2*np.pi*epsilon*sigma**2 * (
                (lambda_p/(2-m)*(sigma/cutoff_val)**(m-2)
                 - 1/(2-n)*(sigma/cutoff_val)**(n-2))
                -
                (lambda_p/(2-m)*(sigma/np.abs(r[mask2]))**(m-2)
                 - 1/(2-n)*(sigma/np.abs(r[mask2]))**(n-2))
            )

            return v

        return V

    # ==============================================================
    # Gaussian soft
    # ==============================================================
    elif pt == "gs":
        def V(r):
            r = np.asarray(r)
            return epsilon * sigma**2 * np.pi * np.exp(-(r/sigma)**2)
        return V

    # ==============================================================
    # Yukawa
    # ==============================================================
    elif pt == "yk":
        kappa = 1.0/sigma
        def V(r):
            r = np.asarray(r)
            out = np.zeros_like(r)
            mask = (r != 0)
            out[mask] = epsilon*np.exp(-kappa*r[mask]) / r[mask]
            return out
        return V

    # ==============================================================
    # Hard core
    # ==============================================================
    elif pt == "hc":
        def V(r):
            r = np.asarray(r)
            v = np.zeros_like(r)
            v[r <= sigma] = 2e16
            return v
        return V

    # ==============================================================
    else:
        raise ValueError(f"Unknown potential type: {pt}")


# =====================================================================
#  SCALAR INTEGRANT (single value)
# =====================================================================

def pair_potential_one_d_integrant(r, specific_pair_potential):
    """
    Scalar version of V(r) for integration.
    """

    pt = specific_pair_potential["type"].lower()
    sigma = specific_pair_potential.get("sigma", 1.0)
    epsilon = specific_pair_potential.get("epsilon", 1.0)
    cutoff = specific_pair_potential.get("cutoff", 5.0)
    EPS = 1e-4

    # ---- zero ----
    if pt in ["zero", "zero_potential"]:
        return 0.0

    # ---- WCA ----
    if pt == "wca":
        rc = 2**(1/6)*sigma
        if r < rc:
            return -epsilon
        elif rc <= r < cutoff:
            return 4*epsilon*((sigma/r)**12 - 2*(sigma/r)**6)
        else:
            return 0.0

    # ---- Mie ----
    if pt == "mie":
        if r < sigma:
            return 0.0
        elif r < cutoff:
            return -4*epsilon*((sigma/r)**48 - (sigma/r)**24)
        else:
            return 0.0

    # ---- MA ----
    if pt == "ma":
        m = float(specific_pair_potential.get("m", 12))
        n = float(specific_pair_potential.get("n", 6))
        lambda_p = float(specific_pair_potential.get("lambda", 1.0))

        r_min = sigma*(lambda_p*m/n)**(1/(m-n))
        factor = m/(m-n)*((m/n)**(n/(m-n)))
        cutoff_val = cutoff

        if abs(r) < r_min:
            return -np.pi*epsilon/lambda_p*(r_min**2 - r**2) + factor*2*np.pi*epsilon*sigma**2 * (
                (lambda_p/(2-m)*(sigma/cutoff_val)**(m-2)
                 - 1/(2-n)*(sigma/cutoff_val)**(n-2))
                -
                (lambda_p/(2-m)*(sigma/r_min)**(m-2)
                 - 1/(2-n)*(sigma/r_min)**(n-2))
            )
        elif abs(r) < cutoff_val:
            return factor*2*np.pi*epsilon*sigma**2 * (
                (lambda_p/(2-m)*(sigma/cutoff_val)**(m-2)
                 - 1/(2-n)*(sigma/cutoff_val)**(n-2))
                -
                (lambda_p/(2-m)*(sigma/abs(r))**(m-2)
                 - 1/(2-n)*(sigma/abs(r))**(n-2))
            )
        else:
            return 0.0

    # ---- Gaussian ----
    if pt == "gs":
        return epsilon * sigma**2 * np.pi * np.exp(-(r/sigma)**2)

    # ---- Yukawa ----
    if pt == "yk":
        if r == 0:
            return 0.0
        return epsilon*np.exp(-r/sigma) / r

    # ---- Hard core ----
    if pt == "hc":
        return 2e16 if r <= sigma else 0.0

    raise ValueError(f"Unknown potential type: {pt}")


# =====================================================================
#  Demo
# =====================================================================

if __name__ == "__main__":
    pot = {"type":"wca", "sigma":1.0, "epsilon":1.0, "cutoff":5.0}

    V = pair_potential_one_d(pot)
    r = np.linspace(0.001, 5, 200)
    print("Vectorized sample:", V(r)[:10])

    print("Scalar sample:", pair_potential_one_d_integrant(1.0, pot))

