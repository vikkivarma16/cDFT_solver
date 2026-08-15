import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Load JSON data
# ---------------------------------
with open("thermo_surface_phi_data.json", "r") as f:
    data = json.load(f)

rho_a = np.array([d["rho_a"] for d in data])
rho_b = np.array([d["rho_b"] for d in data])
mu_a  = np.array([d["pressure"] for d in data])

# ---------------------------------
# Reconstruct grid
# ---------------------------------
unique_rho_a = np.unique(rho_a)
unique_rho_b = np.unique(rho_b)

n_a = len(unique_rho_a)
n_b = len(unique_rho_b)

RHO_A = rho_a.reshape(n_a, n_b)
RHO_B = rho_b.reshape(n_a, n_b)
MU_A  = mu_a.reshape(n_a, n_b)

# ---------------------------------
# 2D Colormap Plot
# ---------------------------------
plt.figure(figsize=(8,6))

# Heatmap
contour = plt.contourf(
    RHO_A,
    RHO_B,
    MU_A,
    levels=100,
    cmap="plasma"
)

# Optional: add contour lines (constant μ)
lines = plt.contour(
    RHO_A,
    RHO_B,
    MU_A,
    levels=15,
    colors="black",
    linewidths=0.5
)

plt.clabel(lines, inline=True, fontsize=8)

plt.xlabel("Density a")
plt.ylabel("Density b")
plt.title("Chemical Potential μ_a(ρ_a, ρ_b)")

cbar = plt.colorbar(contour)
cbar.set_label("Chemical Potential μ_a")

plt.tight_layout()

# ---------------------------------
# Save High Resolution
# ---------------------------------
plt.savefig("mu_a_colormap.png", dpi=600)
plt.close()

print("2D μ_a colormap saved as mu_a_colormap.png")
