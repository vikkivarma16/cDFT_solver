import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Input JSON file
# ============================================================
json_file = "result_wca_gr_comparison.json"
out_dir = Path("plots_wca_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load JSON
# ============================================================
with open(json_file, "r") as f:
    data = json.load(f)

# Radial grid
r = np.array(data["r"])

# Sigmas
sigma_bh = np.array(data["sigma_bh"])
sigma_opt = np.array(data["sigma_opt"])

# r_min for WCA splitting
rmin_bh = {tuple(map(int, k.split(","))): v for k, v in data["rmin_bh"].items()}
rmin_opt = {tuple(map(int, k.split(","))): v for k, v in data["rmin_opt"].items()}

# Attractive & repulsive potentials
u_attr_bh = np.array(data["u_attractive_bh"])
u_rep_bh  = np.array(data["u_repulsive_bh"])
u_attr_opt = np.array(data["u_attractive_opt"])
u_rep_opt  = np.array(data["u_repulsive_opt"])
u_total   = np.array(data["u_total"])

# RDFs
g_wca_bh  = {k: np.array(v) for k, v in data["g_wca_sigma_bh"].items()}
g_wca_opt = {k: np.array(v) for k, v in data["g_wca_sigma_opt"].items()}
g_pred    = {k: np.array(v) for k, v in data["g_pred"].items()}

# ============================================================
# Dimensions
# ============================================================
example_state = next(iter(g_pred))
N = g_pred[example_state].shape[0]
hard_core_pairs = list(rmin_bh.keys())

# ============================================================
# Plot potentials (repulsive & attractive)
# ============================================================
# ============================================================
# 1️⃣ Compare Repulsive with Total
# ============================================================
for (i, j) in hard_core_pairs:
    plt.figure(figsize=(6, 4))
    plt.plot(r, u_total[i, j], lw=2, label="U_total")
    plt.plot(r, u_rep_bh[i, j], "--", lw=2, label="U_rep (σ_BH)")
    plt.plot(r, u_rep_opt[i, j], ":", lw=2, label="U_rep (σ_opt)")
    plt.axvline(sigma_bh[i, j], color="r", ls=":", lw=1.5, label=fr"$\sigma_{{BH}}$")
    plt.axvline(sigma_opt[i, j], color="k", ls="--", lw=1.5, label=fr"$\sigma_{{opt}}$")
    plt.axvline(rmin_bh[(i, j)], color="gray", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{BH}$")
    plt.axvline(rmin_opt[(i, j)], color="blue", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{opt}$")
    plt.xlabel("r")
    plt.ylabel(f"U_{i}{j}(r)")
    plt.title(f"Pair ({i},{j}) - Repulsive vs Total")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"repulsive_vs_total_{i}{j}.png", dpi=600)
    plt.close()

# ============================================================
# 2️⃣ Compare Attractive with Total
# ============================================================
for (i, j) in hard_core_pairs:
    plt.figure(figsize=(6, 4))
    plt.plot(r, u_total[i, j], lw=2, label="U_total")
    plt.plot(r, u_attr_bh[i, j], "--", lw=2, label="U_attr (σ_BH)")
    plt.plot(r, u_attr_opt[i, j], ":", lw=2, label="U_attr (σ_opt)")
    plt.axvline(sigma_bh[i, j], color="r", ls=":", lw=1.5, label=fr"$\sigma_{{BH}}$")
    plt.axvline(sigma_opt[i, j], color="k", ls="--", lw=1.5, label=fr"$\sigma_{{opt}}$")
    plt.axvline(rmin_bh[(i, j)], color="gray", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{BH}$")
    plt.axvline(rmin_opt[(i, j)], color="blue", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{opt}$")
    plt.xlabel("r")
    plt.ylabel(f"U_{i}{j}(r)")
    plt.title(f"Pair ({i},{j}) - Attractive vs Total")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"attractive_vs_total_{i}{j}.png", dpi=600)
    plt.close()

# ============================================================
# Plot g(r) comparisons
# ============================================================
for sname in g_pred:
    print(f"Plotting g(r) comparisons for state: {sname}")
    g_pred_state = g_pred[sname]
    g_bh_state   = g_wca_bh[sname]
    g_opt_state  = g_wca_opt[sname]

    for (i, j) in hard_core_pairs:
        plt.figure(figsize=(6, 4))
        plt.plot(r, g_pred_state[i, j], lw=2, label="g_pred")
        plt.plot(r, g_bh_state[i, j], "--", lw=2, label="g_WCA (σ_BH)")
        plt.plot(r, g_opt_state[i, j], ":", lw=2, label="g_WCA (σ_opt)")

        plt.axvline(sigma_bh[i, j], color="r", ls=":", lw=1.5, label=fr"$\sigma_{{BH}}$")
        plt.axvline(sigma_opt[i, j], color="k", ls="--", lw=1.5, label=fr"$\sigma_{{opt}}$")
        plt.axvline(rmin_bh[(i, j)], color="gray", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{BH}$")
        plt.axvline(rmin_opt[(i, j)], color="blue", ls="-.", lw=1.2, label=r"$r_\mathrm{min}^{opt}$")

        plt.xlabel("r")
        plt.ylabel(f"g_{i}{j}(r)")
        plt.title(f"State: {sname} | Pair ({i},{j})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"g_r_comparison_{sname}_{i}{j}.png", dpi=600)
        plt.close()

print("✅ All WCA potentials and g(r) plots saved in:", out_dir)

