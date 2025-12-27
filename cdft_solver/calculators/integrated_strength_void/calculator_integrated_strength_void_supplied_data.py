import numpy as np
from scipy.fftpack import dst, idst
from collections import defaultdict
from scipy.linalg import solve

# ===============================================================
# DST-based Hankel forward/inverse transforms
# ===============================================================

# ===============================================================

import numpy as np
from pathlib import Path
from collections import defaultdict

from cdft_solver.generators.potential_splitter.generator_potential_splitter_mf import meanfield_potentials
from cdft_solver.generators.potential_splitter.generator_potential_splitter_hc import hard_core_potentials


def build_pair_potential_on_r(r, epsilon, sigma, interaction_type, extra_p=None):
    r = np.asarray(r)
    phi = np.zeros_like(r, dtype=float)
    if interaction_type is None:
        return phi

    interaction_type = interaction_type.lower()
    if interaction_type == "wca":
        r_cut = 5.0 * sigma
        r_min = (2.0**(1/6.0)) * sigma
        mask1 = r < r_min
        mask2 = (r >= r_min) & (r < r_cut)
        phi[mask1] = -epsilon
        phi[mask2] = 4.0 * epsilon * ((sigma / r[mask2])**12 - (sigma / r[mask2])**6)
    elif interaction_type == "ma":
        if extra_p is None:
            extra_p = (12, 6, 1.0)
        m, n, lambda_p = extra_p
        r_min = (lambda_p * m / n)**(1.0 / (m - n))
        cutoff = 10.0 * sigma
        mask1 = r < r_min
        mask2 = (r >= r_min) & (r < cutoff)
        phi[mask1] = -epsilon / lambda_p
        phi[mask2] = (m/(m-n)) * ((m/n)**(n/(m-n))) * epsilon * \
                     (lambda_p * (sigma / r[mask2])**m - (sigma / r[mask2])**n)
    elif interaction_type == "gs":
        phi = epsilon * np.exp(- (r / sigma)**2)
    elif interaction_type == "yk":
        kappa = 1.0 / sigma if sigma != 0 else 1.0
        phi = epsilon * np.exp(-kappa * r) / np.where(r == 0, 1.0, r)
        phi[0] = phi[1]
    elif interaction_type in ["hc", "ghc", "zero"]:
        phi[:] = 0.0
    else:
        phi[:] = 0.0
    return phi


def compute_vk_from_loaded_rdf(species, interaction_defs, r, rdf_data_dir):
    """
    Compute vk matrix from precomputed RDF (g_r) files for all species pairs.

    Parameters
    ----------
    species : list[str]
        List of species names (e.g., ["a", "b", "c"])
    interaction_defs : ndarray
        NxN matrix of lists, each containing dictionaries of potential parameters
    r : ndarray
        Radial grid (must match RDF data grid)
    rdf_data_dir : Path
        Path to folder containing RDF data files for each species pair

    Returns
    -------
    vk : ndarray (N,N)
        Mean-field integral matrix for all species pairs
    """
    N = len(species)
    Nr = len(r)
    phi_r_matrix = np.zeros((N, N, Nr))
    g_r_matrix = np.zeros((N, N, Nr))
    vk = np.zeros((N, N))

    for i, a in enumerate(species):
        for j, b in enumerate(species):
            # Construct filename conventions like "rdf_ab.npz" or "rdf_ab.npy"
            fname_npz = rdf_data_dir / f"rdf_{a}{b}.npz"
            fname_npy = rdf_data_dir / f"rdf_{a}{b}.npy"

            # Load RDF
            if fname_npz.exists():
                data = np.load(fname_npz)
                g_r = data["g_r"] if "g_r" in data else data[list(data.keys())[0]]
            elif fname_npy.exists():
                g_r = np.load(fname_npy)
            else:
                raise FileNotFoundError(f"No RDF data found for pair {a}{b} in {rdf_data_dir}")

            g_r_matrix[i, j, :] = g_r

            # Build total potential for this pair
            plist = interaction_defs[i, j]
            if plist is None:
                continue
            phi_total = np.zeros_like(r)
            for params in plist:
                phi_total += build_pair_potential_on_r(
                    r,
                    params.get("epsilon", 0.0),
                    params.get("sigma", 1.0),
                    params.get("type", "gs"),
                    extra_p=params.get("extra_p")
                )
            phi_r_matrix[i, j, :] = phi_total

            # Integrate to compute v_ij
            integrand = 4.0 * np.pi * r**2 * g_r * phi_total
            vk[i, j] = np.trapz(integrand, r)

    return vk, g_r_matrix, phi_r_matrix

def vk_void_supplied_data(ctx, rdf_data_subfolder="rdf_data"):
    """
    Loads precomputed RDFs for all species pairs and computes vk matrix.

    Parameters
    ----------
    ctx : dict
        Context dictionary with:
            - ctx.scratch_dir: Path to working directory
            - ctx.input_file: Input JSON configuration
    rdf_data_subfolder : str
        Folder (inside installation directory or scratch dir) containing RDF files

    Returns
    -------
    dict with:
        species : list[str]
        vk : ndarray (N,N)
        g_r_matrix : ndarray (N,N,len(r))
        r : ndarray
    """
    # Load potential and hard-core data
    pot_data = meanfield_potentials(ctx, mode="meanfield")
    hc_data = hard_core_potentials(ctx)
    species = pot_data["species"]
    N = len(species)

    # Assemble all potentials from all levels (primary, secondary, etc.)
    pair_contribs = defaultdict(list)
    for level, pairs in pot_data["interactions"].items():
        for pair, params in pairs.items():
            a, b = pair[0], pair[1]
            pair_contribs[(a, b)].append(params)

    interaction_defs = np.empty((N, N), dtype=object)
    for i, a in enumerate(species):
        for j, b in enumerate(species):
            plist = []
            if (a, b) in pair_contribs:
                plist.extend(pair_contribs[(a, b)])
            if (b, a) in pair_contribs and (b != a):
                plist.extend(pair_contribs[(b, a)])
            interaction_defs[i, j] = plist if plist else None

    # Path to RDF data
    rdf_data_dir = Path(ctx.get("rdf_dir", Path.cwd() / "rdf"))

    if not rdf_data_dir.exists():
        raise FileNotFoundError(f"RDF data directory not found: {rdf_data_dir}")


    # Load sample r-grid (assumed consistent for all pairs)
    r_file = rdf_data_dir / "r.npy"
    if not r_file.exists():
        raise FileNotFoundError("Missing r.npy file for radial grid in RDF folder")
    r = np.load(r_file)

    # Compute vk from RDFs
    vk, g_r_matrix, phi_r_matrix = compute_vk_from_loaded_rdf(
        species, interaction_defs, r, rdf_data_dir
    )

    return {"species": species, "vk": vk}


