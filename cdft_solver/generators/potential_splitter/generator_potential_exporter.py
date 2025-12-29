import numpy as np
from pathlib import Path
from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi

def export_pair_potential(ctx, key, inter, grid_points=5000, filename_prefix="potential_"):
    """
    Export pair potential for a given interaction.

    Parameters
    ----------
    key : str
        Pair key (e.g., "aa", "ab").
    inter : dict
        Interaction dictionary with at least 'type' and 'sigma' (optional 'cutoff').
    output_dir : str
        Directory where to save the file.
    grid_points : int
        Number of points in the r-grid.
    filename_prefix : str
        Prefix for output file name.
    """
    # Create output directory
    out_path = Path(ctx.scratch_dir) 
    #out_path.mkdir(parents=True, exist_ok=True)

    #out_path.mkdir(parents=True, exist_ok=True)

    # Determine grid
    cutoff = inter.get("cutoff", inter.get("sigma", 1.0) * 5.0)
    r_values = np.linspace(1e-5, cutoff, grid_points)

    # Generate potential
    V_func = ppi(inter)
    u_values = V_func(r_values)

    # Save file
    filename = out_path / f"{filename_prefix}{key}.txt"
    np.savetxt(filename, np.column_stack([r_values, u_values]), header="r U(r)")
    print(f"✅ Potential exported: {filename}")

