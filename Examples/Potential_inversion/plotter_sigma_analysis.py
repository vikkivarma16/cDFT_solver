import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Inputs
# ============================================================

json_file = "result_sigma_analysis.json"
out_dir   = Path("plots_sigma_repulsive")
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load JSON
# ============================================================

with open(json_file, "r") as f:
    data = json.load(f)

# Radial grid
r = np.array(data["r"])

# Sigmas
sigma_opt = np.array(data["sigma_opt"])
sigma_bh  = np.array(data["sigma_bh"])

# BH metadata
bh_meta = {
    tuple(map(int, k.split(","))): v
    for k, v in data["bh_meta"].items()
}

# RDFs
g_target = {k: np.array(v) for k, v in data["g_target"].items()}
g_ref    = {k: np.array(v) for k, v in data["g_ref_wca"].items()}
g_rep_opt = {k: np.array(v) for k, v in data["g_rep_sigma_opt"].items()}
g_rep_bh  = {k: np.array(v) for k, v in data["g_rep_sigma_bh"].items()}


# ============================================================
# Dimensions
# ============================================================

example_state = next(iter(g_target))
N = g_target[example_state].shape[0]


# ============================================================
# Plot diagnostics
# ============================================================

for sname in g_target:

    print(f"Plotting sigma diagnostics for state: {sname}")

    g_tgt = g_target[sname]
    g_r   = g_ref[sname]
    g_o   = g_rep_opt[sname]
    g_b   = g_rep_bh[sname]

    for (i, j), meta in bh_meta.items():

        r0 = meta["r0"]

        plt.figure(figsize=(6, 4))

        plt.plot(r, g_tgt[i, j], lw=2, label="g_target")
        plt.plot(r, g_r[i, j], "--", lw=2, label="g_ref (WCA rep)")
        plt.plot(r, g_o[i, j], ":", lw=2, label="g_rep (σ_opt)")
        plt.plot(r, g_b[i, j], "-.", lw=2, label="g_rep (σ_BH)")

        # Mark characteristic lengths
        plt.axvline(
            sigma_opt[i, j],
            color="k",
            ls="--",
            lw=1,
            label=fr"$\sigma_{{opt}}={sigma_opt[i,j]:.3f}$"
        )

        plt.axvline(
            sigma_bh[i, j],
            color="r",
            ls=":",
            lw=1,
            label=fr"$d_{{BH}}={sigma_bh[i,j]:.3f}$"
        )

        plt.axvline(
            r0,
            color="gray",
            ls="-.",
            lw=1,
            label=r"$r_0$ (u=0)"
        )

        plt.xlabel("r")
        plt.ylabel(f"g$_{{{i}{j}}}$(r)")

        plt.title(
            f"State: {sname} | Pair ({i},{j})"
        )

        plt.legend(fontsize=9)
        plt.tight_layout()

        fname = out_dir / f"sigma_repulsive_{sname}_{i}{j}.png"
        plt.savefig(fname, dpi=600)
        plt.close()


print("✅ Sigma / repulsive diagnostic plots saved to:", out_dir)

