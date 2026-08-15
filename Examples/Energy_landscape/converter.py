import numpy as np

# ============================================================
# Load data
# ============================================================

data = np.loadtxt("polymer_coexistence_scan.txt")

# Handle optional header
if data.shape[1] != 10:
    data = np.loadtxt("polymer_coexistence_scan.txt", skiprows=1)

# ============================================================
# Output container
# ============================================================

converted = []

# ============================================================
# Loop over rows
# ============================================================

for row in data:

    # --------------------------------------------------------
    # Input columns
    # --------------------------------------------------------
    rho_total = row[0]
    rho_a_in = row[1]
    rho_b_in = row[2]

    # Phase data
    p1 = np.array([row[3], row[4], row[5]])  # phase 1 raw
    p2 = np.array([row[6], row[7], row[8]])  # phase 2 raw

    f1 = row[9]
    f2 = 1.0 - f1

    # --------------------------------------------------------
    # Phase ordering condition: rho_c dominance
    # --------------------------------------------------------

    if p2[2] > p1[2]:
        # swap so phase1 always has higher rho_c
        p1, p2 = p2, p1
        f1, f2 = f2, f1

    # --------------------------------------------------------
    # Recompute total densities
    # --------------------------------------------------------

    rho_t1 = np.sum(p1)
    rho_t2 = np.sum(p2)

    # --------------------------------------------------------
    # Compute composition variables
    # --------------------------------------------------------

    x1 = p1[0] / (p1[0] + p1[1]) if (p1[0] + p1[1]) > 0 else 0.0
    x2 = p2[0] / (p2[0] + p2[1]) if (p2[0] + p2[1]) > 0 else 0.0

    y1 = p1[2] / rho_t1 if rho_t1 > 0 else 0.0
    y2 = p2[2] / rho_t2 if rho_t2 > 0 else 0.0

    # --------------------------------------------------------
    # Store cleaned result
    # --------------------------------------------------------

    converted.append([
        rho_total/2,
        f1,
        rho_t1,
        rho_t2,
        x1,
        x2,
        y1,
        y2
    ])

converted = np.array(converted)

# ============================================================
# Save output
# ============================================================

header = (
    "rho_total "
    "fraction_phase1 "
    "rho_t_phase1 rho_t_phase2 "
    "x_phase1 x_phase2 "
    "y_phase1 y_phase2"
)

np.savetxt(
    "polymer_coexistence_scan_sorted.txt",
    converted,
    fmt="%.8f",
    header=header
)

print("Done → sorted by rho_c dominance in phase 1")
