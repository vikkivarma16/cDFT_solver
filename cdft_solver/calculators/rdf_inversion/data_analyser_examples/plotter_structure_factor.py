import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# Configuration
# ===============================
json_file = "result_sigma_analysis.json"
out_dir = Path("structure_factor_all_pairs")
out_dir.mkdir(exist_ok=True)

# Species densities
rho = {
    0: 0.09,  # colloid
    1: 0.27   # polymer
}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
})

# ===============================
# Load data
# ===============================
with open(json_file, "r") as f:
    data = json.load(f)

r = np.array(data["r"])

g_real = {k: np.array(v) for k, v in data["g_real"].items()}
g_opt  = {k: np.array(v) for k, v in data["g_rep_sigma_opt"].items()}
g_bh   = {k: np.array(v) for k, v in data["g_rep_sigma_bh"].items()}

example_state = next(iter(g_real))
N = g_real[example_state].shape[0]

# All unique pairs (cc, cp, pp)
pairs = [(0, 0), (0, 1), (1, 1)]

# ===============================
# Structure factor function (mixture)
# ===============================
def structure_factor_ij(k, r, g_ij, rho_i, rho_j):
    """
    Partial structure factor S_ij(k)
    """
    kr = np.outer(k, r)
    sinc = np.sin(kr) / kr
    sinc[:, r == 0] = 1.0

    integrand = (g_ij - 1.0) * r**2
    integral = np.trapz(integrand * sinc, r, axis=1)

    delta = 1.0 if rho_i == rho_j else 0.0

    return delta + 4 * np.pi * np.sqrt(rho_i * rho_j) * integral

# ===============================
# k grid
# ===============================
k = np.linspace(0.05, 30.0, 600)

# ===============================
# Plot: 3 × 1 (cc, cp, pp)
# ===============================
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

pair_labels = {
    (0, 0): "cc",
    (0, 1): "cp",
    (1, 1): "pp"
}

letters = ["(a)", "(b)", "(c)"]

for idx, (i, j) in enumerate(pairs):
    ax = axs[idx]

    rho_i = rho[i]
    rho_j = rho[j]

    # Compute structure factors
    S_real = structure_factor_ij(
        k, r, g_real[example_state][i, j], rho_i, rho_j
    )
    S_opt = structure_factor_ij(
        k, r, g_opt[example_state][i, j], rho_i, rho_j
    )
    S_bh = structure_factor_ij(
        k, r, g_bh[example_state][i, j], rho_i, rho_j
    )

    # Plot
    ax.plot(k, S_real, lw=4, color="black", label=r"$S_{\rm real}$")
    ax.plot(k, S_opt, lw=4, color="blue", ls=":", label=r"$S_{\sigma_{\rm opt}}$")
    ax.plot(k, S_bh, lw=4, color="red", ls="-.", label=r"$S_{\sigma_{\rm BH}}$")

    ax.set_ylabel(
        rf"$S_{{{pair_labels[(i,j)]}}}(k)$",
        fontsize=18
    )
    ax.set_title(
        rf"\rm {letters[idx]}",
        loc="center",
        pad=-10,
        fontsize=17
    )

    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(0, 30)

    if idx == 0:
        ax.legend(
            fontsize=15,
            ncol=2,
            frameon=True,
            loc="lower right",
            columnspacing=0.4,
            handletextpad=0.3
        )

# X-label only on bottom
axs[-1].set_xlabel(r"$k$", fontsize=18)
axs[-1].tick_params(axis="x", labelsize=16)

# ===============================
# Save
# ===============================
plt.tight_layout()
plt.savefig(out_dir / "structure_factor_all_pairs_3x1.png", dpi=800, bbox_inches="tight")
plt.close(fig)

print("✅ Structure factors S_ij(k) for cc, cp, pp saved →", out_dir)

