import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Load φ-grid dataset
# ---------------------------------
with open("thermo_surface_phi_data.json", "r") as f:
    data = json.load(f)

phi_a = np.array([d["phi_a"] for d in data])
phi_b = np.array([d["phi_b"] for d in data])
mu_a  = np.array([d["mu_a"] for d in data])
mu_b  = np.array([d["mu_b"] for d in data])

# ---------------------------------
# Reconstruct structured grid
# ---------------------------------
unique_phi_a = np.unique(phi_a)
unique_phi_b = np.unique(phi_b)

n_a = len(unique_phi_a)
n_b = len(unique_phi_b)

PHI_A = phi_a.reshape(n_a, n_b)
PHI_B = phi_b.reshape(n_a, n_b)
MU_A  = mu_a.reshape(n_a, n_b)
MU_B  = mu_b.reshape(n_a, n_b)

# ---------------------------------
# Plot contour lines only
# ---------------------------------
plt.figure(figsize=(8,6))

# μ_a contours (solid)
levels_a = np.linspace(np.min(MU_A), np.max(MU_A), 12)
contour_a = plt.contour(
    PHI_A,
    PHI_B,
    MU_A,
    levels=levels_a,
    linewidths=1.5,
)

# μ_b contours (dashed)
levels_b = np.linspace(np.min(MU_B), np.max(MU_B), 12)
contour_b = plt.contour(
    PHI_A,
    PHI_B,
    MU_B,
    levels=levels_b,
    linestyles='dashed',
    linewidths=1.2,
)

plt.clabel(contour_a, inline=True, fontsize=8)
plt.clabel(contour_b, inline=True, fontsize=8)

plt.xlabel("Packing Fraction φ_a")
plt.ylabel("Packing Fraction φ_b")
plt.title("Constant μ_a (solid) and μ_b (dashed) Contours")

plt.tight_layout()

# ---------------------------------
# Save high resolution PNG
# ---------------------------------
plt.savefig("mu_a_mu_b_contours_phi_space.png", dpi=600)
plt.close()

print("Dual chemical potential contour plot saved.")
