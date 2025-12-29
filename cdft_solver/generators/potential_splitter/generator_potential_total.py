import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def total_pair_potentials(
    ctx,
    hc_prefix="supplied_data_potential_hc_",
    mf_prefix="supplied_data_potential_mf_",
    out_prefix="supplied_data_potential_total_",
    plot=False
):
    """
    Build TOTAL pair potentials by adding:
        U_total(r) = U_hc(r) + U_mf(r)

    Inputs are read from ctx.scratch_dir.
    Outputs are written to the same directory.

    Rules
    -----
    • Missing HC or MF contribution → treated as zero
    • Different grids are interpolated to the HC grid
    • Optional plotting
    """

    scratch = Path(ctx.scratch_dir)

    hc_files = list(scratch.glob(f"{hc_prefix}*.txt"))
    mf_files = list(scratch.glob(f"{mf_prefix}*.txt"))

    # Collect all pair keys
    def extract_key(path, prefix):
        return path.stem.replace(prefix, "")

    pairs = set()
    pairs |= {extract_key(f, hc_prefix) for f in hc_files}
    pairs |= {extract_key(f, mf_prefix) for f in mf_files}

    if not pairs:
        print("⚠️ No HC or MF potentials found — nothing to combine.")
        return

    # ------------------------------------------------------
    # Combine potentials pairwise
    # ------------------------------------------------------
    for key in sorted(pairs):
        hc_file = scratch / f"{hc_prefix}{key}.txt"
        mf_file = scratch / f"{mf_prefix}{key}.txt"

        r_total = None
        u_total = None

        # Load HC
        if hc_file.exists():
            data = np.loadtxt(hc_file)
            r_total = data[:, 0]
            u_total = data[:, 1].copy()
        else:
            u_total = None

        # Load MF
        if mf_file.exists():
            data = np.loadtxt(mf_file)
            r_mf = data[:, 0]
            u_mf = data[:, 1]

            if r_total is None:
                r_total = r_mf
                u_total = u_mf.copy()
            else:
                # Interpolate MF onto HC grid
                if not np.allclose(r_total, r_mf):
                    interp = interp1d(r_mf, u_mf, kind='linear', fill_value="extrapolate")
                    u_total += interp(r_total)
                else:
                    u_total += u_mf

        # Write total potential
        out_file = scratch / f"{out_prefix}{key}.txt"
        np.savetxt(out_file, np.column_stack([r_total, u_total]), header="r U_total(r)")
        print(f"✅ Exported total potential: {out_file}")

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(r_total, u_total, label=f'Total {key}')
            plt.xlabel("r")
            plt.ylabel("U_total(r)")
            plt.title(f"Total potential for pair {key}")
            plt.ylim(-5, 5)  # Set y-axis limits from -5 to 5
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_file = scratch / f"total_potential_for_pair_{key}.png"
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"✅ Total potential Plot exported for verification purpose: {plot_file}")


