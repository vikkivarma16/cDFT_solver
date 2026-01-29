import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===============================
# Configuration
# ===============================
delta_c_file = "delta_c_by_state.json"
gr_file      = "result_G_of_r.json"
out_dir      = Path("delta_c_Gu_CP")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",
})

# ===============================
# Load data
# ===============================
with open(delta_c_file, "r") as f:
    delta_c_data = json.load(f)

with open(gr_file, "r") as f:
    gr_data = json.load(f)

r = np.array(delta_c_data["r"])
example_state = list(delta_c_data["delta_c_real_ref"].keys())[0]

# -------------------------------
# Δc(r) with surface factor
# -------------------------------
delta_c_real_ref_sf       = 4 * np.pi * r**2 * np.array(delta_c_data["delta_c_real_ref"][example_state])
delta_c_sigma_opt_sigma_opt_sf = 4 * np.pi * r**2 * np.array(delta_c_data["delta_c_sigma_opt_sigma_opt"][example_state])

# Δc(r) without surface factor
delta_c_real_ref = np.array(delta_c_data["delta_c_real_ref"][example_state])
delta_c_sigma_opt_sigma_opt = np.array(delta_c_data["delta_c_sigma_opt_sigma_opt"][example_state])

# -------------------------------
# G_u(r) with surface factor
# -------------------------------
G_u_r_real_sf      = 4 * np.pi * r**2 * np.array(gr_data["G_u_r_real"][example_state])
G_u_r_sigma_opt_sf = 4 * np.pi * r**2 * np.array(gr_data["G_u_r_sigma_opt"][example_state])

# G_u(r) without surface factor
G_u_r_real = np.array(gr_data["G_u_r_real"][example_state])
G_u_r_sigma_opt = np.array(gr_data["G_u_r_sigma_opt"][example_state])

# -------------------------------
# u_attractive with surface factor
# -------------------------------
u_attractive_sf = 4 * np.pi * r**2 * np.array(gr_data["u_attractive_real"])
u_attractive = np.array(gr_data["u_attractive_real"])  # for inset

# -------------------------------
# Colloid-polymer pair
# -------------------------------
i, j = 0, 1

# ===============================
# Function to find common crossing point
# ===============================
def find_common_crossing(curves, r, r_guess=1.0):
    idx_guess = np.argmin(np.abs(r - r_guess))
    window = 5
    idx_range = np.arange(max(idx_guess-window, 0), min(idx_guess+window+1, len(r)))
    for idx in idx_range:
        signs = [np.sign(curve[i,j,idx]) for curve in curves]
        if all(s == signs[0] for s in signs):
            continue
        else:
            return r[idx]
    return r[idx_guess]

# -------------------------------
# Curves list for crossing detection
# -------------------------------
curves_sf = [
    delta_c_real_ref_sf,
    delta_c_sigma_opt_sigma_opt_sf,
    G_u_r_real_sf,
    G_u_r_sigma_opt_sf
]

# Crossing point (lower limit) and upper limit
r_lower = find_common_crossing(curves_sf, r, r_guess=1.0)
r_upper = 3.5
idx_lower = np.searchsorted(r, r_lower)
idx_upper = np.searchsorted(r, r_upper)

# ===============================
# Compute integrals
# ===============================
integrals = {
    "Δc_real_ref": np.trapz(delta_c_real_ref_sf[i,j,idx_lower:idx_upper+1], r[idx_lower:idx_upper+1]),
    "Δc_sigma_sigma": np.trapz(delta_c_sigma_opt_sigma_opt_sf[i,j,idx_lower:idx_upper+1], r[idx_lower:idx_upper+1]),
    "G_real": np.trapz(G_u_r_real_sf[i,j,idx_lower:idx_upper+1], r[idx_lower:idx_upper+1]),
    "G_sigma": np.trapz(G_u_r_sigma_opt_sf[i,j,idx_lower:idx_upper+1], r[idx_lower:idx_upper+1]),
    "u_attractive": np.trapz(u_attractive_sf[i,j,idx_lower:idx_upper+1], r[idx_lower:idx_upper+1])
}

integral_text = (
    rf"$\int s^* \Delta c^{{\rm real-ref}}_{{ cp}} dr = {integrals['Δc_real_ref']:.2f}$" "\n"
    rf"$\int s^* \Delta c^{{\rm \sigma-\sigma}}_{{ cp}} dr = {integrals['Δc_sigma_sigma']:.2f}$" "\n"
    rf"$\int s^* \phi_{{ cp}} ^{{\rm p}} G^{{\rm real}}_{{ cp}} dr = {integrals['G_real']:.2f}$" "\n"
    rf"$\int s^* \phi_{{ cp}} ^{{\rm p}} G^{{\rm \sigma}}_{{ cp}} dr = {integrals['G_sigma']:.2f}$" "\n"
    rf"$\int s^* \phi_{{ cp}} ^{{\rm p}} dr = {integrals['u_attractive']:.2f}$"
)

# ===============================
# Main plot
# ===============================
fig, ax = plt.subplots(figsize=(8, 6))


ax.plot(r, G_u_r_sigma_opt_sf[i,j,:], lw=4, color="blue", alpha=0.5, label=r"$s^* \phi_{cp}^ {\rm p} G_{ cp}^{\rm \sigma}$")
ax.plot(r, G_u_r_real_sf[i,j,:], lw=4,  ls="--", color="black", alpha=0.6, label=r"$s^* \phi_{cp}^ {\rm p} G_{ cp}^{\rm real}$")
ax.plot(r, delta_c_real_ref_sf[i,j,:], lw=4, color="black", label=r"$s^* \Delta c_{ cp}^{\rm real-ref}$")
ax.plot(r, delta_c_sigma_opt_sigma_opt_sf[i,j,:], lw=4, ls="-.", color="green", label=r"$s^* \Delta c_{ cp}^{\rm \sigma-\sigma}$", alpha = 0.9)
ax.plot(r, u_attractive_sf[i,j,:], lw=4, ls=":", color="red", alpha=0.7, label=r"$s^* \phi_{{cp}} ^{{\rm p}}$")

# Vertical dashed lines (no legend)
ax.axvline(r_lower, color='black', lw=1, ls='--', label='_nolegend_')
ax.axvline(r_upper, color='black', lw=1, ls='--', label='_nolegend_')

# Axis settings
ax.set_xlim(0.0, 10)
ax.set_ylim(-12, 1)
ax.set_yticks([-10, 0])
ax.set_ylabel(r"$s^* \Delta c_{ cp},\ s^* \phi_{ cp}^ {\rm p} G_{ cp},\ s^* \phi_{ cp}^ {\rm p}$", fontsize=19, labelpad=-25)
ax.set_xlabel(r"$r$", fontsize=19)
ax.set_xticks([0, 1, 2, 3, 4])
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.legend(fontsize=17, ncol=2, frameon=True, columnspacing=0.4, handletextpad=0.3, loc='lower right', framealpha=1.0)

# Integral text
ax.text(0.4, 0.04, integral_text, transform=ax.transAxes,
        fontsize=17, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1.0))

# ===============================
# Inset: without surface factor
# ===============================
ax_inset = inset_axes(ax, width="30%", height="30%", bbox_to_anchor=(0.53, 0.4, 1.5, 1.4),
                      bbox_transform=ax.transAxes, loc='lower left')

ax_inset.plot(r, G_u_r_sigma_opt[i,j,:], lw=4, color="blue", alpha=0.5)
ax_inset.plot(r, G_u_r_real[i,j,:], lw=4, ls="--", color="black", alpha=0.6)
ax_inset.plot(r, delta_c_real_ref[i,j,:], lw=4, color="black")
ax_inset.plot(r, delta_c_sigma_opt_sigma_opt[i,j,:], lw=4, ls="-.", color="green", alpha = 0.9)
ax_inset.plot(r, u_attractive[i,j,:], lw=4, ls=":", color="red", alpha=0.7)

ax_inset.set_xlim(0.0, 10)
ax_inset.set_ylim(-0.5, 0.2)
ax_inset.set_xticks([0, 1, 2, 3, 4])
ax_inset.set_yticks([-0.5, 0])
ax_inset.tick_params(axis="x", labelsize=14)
ax_inset.tick_params(axis="y", labelsize=14)

# ===============================
# Save figure
# ===============================
plt.tight_layout()
plt.savefig(out_dir / "delta_c_Gu_CP_inset_integrals_crossing_u_attractive.png", dpi=600, bbox_inches="tight")
plt.close(fig)

print("✅ Δc(r), G_u(r), u_attractive plotted for CP pair with inset, integrals, and crossing points →", out_dir)

