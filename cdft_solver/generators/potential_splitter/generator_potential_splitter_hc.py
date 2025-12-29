import json
import numpy as np
from pathlib import Path
from scipy.integrate import simpson

# Import your existing potential generator
from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi


def hard_core_potentials(
    ctx,
    beta=1.0,
    Nr=5000,
    hc_json_name="potential_parameter_hc.json"
):
    """
    Hard-core detection and consistency module.

    Rules implemented:
    ------------------
    • Hard-core detection is PAIR-BASED
    • σ_ij known OR (σ_ii AND σ_jj known) → consistent
    • Additive rule used only when exactly one unknown
    • If no HC exists anywhere → trivial pass (all σ=0, flag=0)
    • Returns NOTHING; writes hc_parameter.json only
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def barker_henderson_diameter(r, u):
        integrand = 1.0 - np.exp(-beta * np.clip(u, -100, 100))
        return simpson(integrand, r)

    def is_hard_core_type(ptype):
        return ptype.lower() in {"hc", "ghc", "hardcore", "lj", "mie"}

    def load_interaction_data(ctx):
        scratch = Path(ctx.scratch_dir)
        json_path = scratch / "input_data_particles_interactions_parameters.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing interaction file in {scratch}")
        with open(json_path) as f:
            data = json.load(f)
        return data["particles_interactions_parameters"]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    params = load_interaction_data(ctx)
    species = params["species"]
    interactions = params["interactions"]
    levels = ["primary", "secondary", "tertiary"]

    n = len(species)
    idx = {sp: i for i, sp in enumerate(species)}

    sigma = np.zeros((n, n), dtype=float)
    flag = np.zeros((n, n), dtype=int)
    explicit = np.zeros((n, n), dtype=bool)

    # ------------------------------------------------------------------
    # STEP 1: Detect explicit hard cores (pair-based)
    # ------------------------------------------------------------------
    for level in levels:
        for key, inter in interactions.get(level, {}).items():
            sp_i, sp_j = key[0], key[1]
            i, j = idx[sp_i], idx[sp_j]

            ptype = inter["type"].lower()
            if not is_hard_core_type(ptype):
                continue

            # Direct hard-core
            if ptype in {"hc", "ghc", "hardcore"}:
                s = inter.get("sigma", 0.0)
                sigma[i, j] = sigma[j, i] = s
                
                ptype in {"hc", "hardcore"}:
                    flag [i, j] = 1
                explicit[i, j] = explicit[j, i] = True
                continue

            # Soft-core → detect via potential
            pot = ppi(inter.copy())
            r = np.linspace(1e-5, inter.get("sigma", 1.0), Nr)
            u = np.clip(pot(r), -1e3, 1e7)

            if np.any(u[:5] > 1e6):
                s = barker_henderson_diameter(r, u)
                sigma[i, j] = sigma[j, i] = s
                explicit[i, j] = explicit[j, i] = True

    # ------------------------------------------------------------------
    # STEP 2: Trivial pass if no hard-core exists anywhere
    # ------------------------------------------------------------------
   

    # ------------------------------------------------------------------
    # STEP 3: Consistency check (OR logic)
    # ------------------------------------------------------------------

                    
    for i in range(n):
        j=i
        if explicit[i, j]:
            continue

        found = False
        for k in range(n):
            if explicit[i, k] and explicit[k, k]:
                sigma[i, i] = 2 * sigma[i, k] - sigma[k, k]  # reverse additive rule
                explicit[i, i] = True
                found = True
                break
        if not found:
            raise ValueError(f"Cannot determine σ({species[i]},{species[j]})")
        
                    
    for i in range(n):
        for j in range(i+1, n):
            # Off-diagonal additive rule
            if explicit[i, i] and explicit[j, j]:
                sigma[i, j] = sigma[j, i] = 0.5 * (sigma[i, i] + sigma[j, j])
                explicit[i, j] = explicit[j, i] = True
            else:
                raise ValueError(f"Cannot determine σ({species[i]},{species[j]})")



    # ------------------------------------------------------------------
    # STEP 6: Write JSON output
    # ------------------------------------------------------------------
    hc_data = {
        "species": species,
        "sigma": sigma.tolist(),
        "flag": flag.tolist(),
    }

    out = Path(ctx.scratch_dir) / hc_json_name
    with open(out, "w") as f:
        json.dump(hc_data, f, indent=2)
        
        
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # STEP 7: Reassign interactions as HC and export potentials in-house
    # ------------------------------------------------------------------
    for i in range(n):
        for j in range(i, n):
            sp_i, sp_j = species[i], species[j]
            s = sigma[i, j]
            f = int(flag[i, j])

            # Determine cutoff: take from original interaction if exists, else 5*sigma
            cutoff = None
            for level in levels:
                key = f"{sp_i}{sp_j}"
                if key in interactions[level]:
                    cutoff = interactions[level][key].get("cutoff")
                    break
            if cutoff is None:
                cutoff = s * 5.0

            # Build r-grid
            r_values = np.linspace(1e-5, cutoff, 1000)

            # Build HC potential: hard-core → step function or just 0 beyond sigma
            # Here we set U(r) = 0 beyond sigma, very large (1e6) below sigma
            u_values = np.where(r_values < s, 1e6, 0.0)

            # Prepare filename
            filename_prefix = "supplied_data_potential_hc_"
            out_path = Path(ctx.scratch_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            filename = out_path / f"{filename_prefix}{sp_i}{sp_j}.txt"

            # Save potential to file
            np.savetxt(filename, np.column_stack([r_values, u_values]), header="r U(r)")
            print(f"✅ Exported HC potential: {filename}")




