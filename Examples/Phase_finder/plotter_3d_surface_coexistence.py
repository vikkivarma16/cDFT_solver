import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

# Interpolator for μ_a (for coloring)
interp_mu_a = RegularGridInterpolator(
    (unique_phi_a, unique_phi_b),
    MU_A
)

# ---------------------------------
# Plot contours
# ---------------------------------
plt.figure(figsize=(8,6))

levels_a = np.linspace(np.min(MU_A), np.max(MU_A), 12)
contour_a = plt.contour(
    PHI_A,
    PHI_B,
    MU_A,
    levels=levels_a,
    linewidths=1.5,
)

levels_b = np.linspace(np.min(MU_B), np.max(MU_B), 12)
contour_b = plt.contour(
    PHI_A,
    PHI_B,
    MU_B,
    levels=levels_b,
    linestyles='dashed',
    linewidths=1.2,
)

# ---------------------------------
# Line segment intersection function
# ---------------------------------
def segment_intersection(p1, p2, p3, p4):
    """
    Return intersection point of segments p1-p2 and p3-p4 if exists.
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-12:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

    # Check if within both segments
    if (
        min(x1,x2)-1e-12 <= px <= max(x1,x2)+1e-12 and
        min(y1,y2)-1e-12 <= py <= max(y1,y2)+1e-12 and
        min(x3,x4)-1e-12 <= px <= max(x3,x4)+1e-12 and
        min(y3,y4)-1e-12 <= py <= max(y3,y4)+1e-12
    ):
        return px, py

    return None

# ---------------------------------
# Find all contour intersections
# ---------------------------------
intersections = []

for col_a in contour_a.collections:
    for path_a in col_a.get_paths():
        verts_a = path_a.vertices

        for col_b in contour_b.collections:
            for path_b in col_b.get_paths():
                verts_b = path_b.vertices

                for i in range(len(verts_a)-1):
                    p1 = verts_a[i]
                    p2 = verts_a[i+1]

                    for j in range(len(verts_b)-1):
                        p3 = verts_b[j]
                        p4 = verts_b[j+1]

                        pt = segment_intersection(p1, p2, p3, p4)
                        if pt is not None:
                            intersections.append(pt)

# Remove duplicates
intersections = np.unique(np.array(intersections), axis=0)

# ---------------------------------
# Color intersections by μ_a value
# ---------------------------------
mu_values = interp_mu_a(intersections)

sc = plt.scatter(
    intersections[:,0],
    intersections[:,1],
    c=mu_values,
    cmap="viridis",
    s=40,
    edgecolors="black",
    zorder=5
)

plt.colorbar(sc, label="μ_a value at intersection")

plt.xlabel("Packing Fraction φ_a")
plt.ylabel("Packing Fraction φ_b")
plt.title("Iso-μ_a / Iso-μ_b Intersections")

plt.tight_layout()
plt.savefig("iso_mu_intersections.png", dpi=600)
plt.close()

print("Intersection map saved.")
