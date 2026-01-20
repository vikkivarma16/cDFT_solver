import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Load JSON file
# ===============================
json_file = "result_attractive_calibration.json"
with open(json_file, "r") as f:
    data = json.load(f)

r = np.array(data["r"])
sigma_opt = np.array(data["sigma_opt"])
sigma_bh  = np.array(data["sigma_bh"])

g_pred = {k: np.array(v) for k, v in data["g_pred"].items()}  # keyed by state

sigma_opt_results = data["sigma_opt_results"]
sigma_bh_results  = data["sigma_bh_results"]

# Potentials
u_attr_opt = np.array(sigma_opt_results["u_attractive"])
u_total_opt = np.array(sigma_opt_results["u_total"])

u_attr_bh  = np.array(sigma_bh_results["u_attractive"])
u_total_bh = np.array(sigma_bh_results["u_total"])

# g(r) from UR
g_ur_opt = {k: np.array(v) for k, v in sigma_opt_results["g_ur"].items()}
g_ur_bh  = {k: np.array(v) for k, v in sigma_bh_results["g_ur"].items()}

out_dir = Path("./attractive_analysis_plots")
out_dir.mkdir(parents=True, exist_ok=True)

N = u_total_opt.shape[0]

# ===============================
# Loop over all states
# ===============================
for sname in g_pred.keys():
    g_pred_state = g_pred[sname]
    g_ur_state_opt = g_ur_opt[sname]
    g_ur_state_bh  = g_ur_bh[sname]

    # ---------------------------
    # Loop over all hard-core pairs
    # ---------------------------
    for i in range(N):
        for j in range(i, N):

            # --------- Potential comparison ---------
            plt.figure(figsize=(6, 4))
            plt.plot(r, u_total_opt[i, j], lw=2, label="U_total (σ_opt)")
            plt.plot(r, u_attr_opt[i, j], "--", lw=2, label="U_attr (σ_opt)")
            plt.plot(r, u_total_bh[i, j], lw=2, label="U_total (σ_BH)")
            plt.plot(r, u_attr_bh[i, j], "--", lw=2, label="U_attr (σ_BH)")
            plt.xlabel("r")
            plt.ylabel(f"U_{i}{j}(r)")
            plt.title(f"Pair ({i},{j}) - Attractive vs Total")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sname}_attractive_vs_total_{i}{j}.png", dpi=600)
            plt.close()

            # --------- g(r) comparison ---------
            plt.figure(figsize=(6, 4))
            plt.plot(r, g_pred_state[i, j], lw=2, label="g_pred")
            plt.plot(r, g_ur_state_opt[i, j], "--", lw=2, label="g_ur (σ_opt)")
            plt.plot(r, g_ur_state_bh[i, j], ":", lw=2, label="g_ur (σ_BH)")
            plt.xlabel("r")
            plt.ylabel(f"g_{i}{j}(r)")
            plt.title(f"State: {sname} | Pair ({i},{j}) - g(r) comparison")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{sname}_g_r_comparison_{i}{j}.png", dpi=600)
            plt.close()

print("✅ Attractive calibration analysis plots saved in:", out_dir)

