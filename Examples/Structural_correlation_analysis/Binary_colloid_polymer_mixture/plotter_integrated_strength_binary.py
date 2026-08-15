import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ===============================
# GLOBAL STYLE
# ===============================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble":
        r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",

    # Thicker axes and ticks
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})


# ===============================
# COLOR AND LINE SETTINGS
# ===============================
COLOR_TI_REAL = "#000000"    # Black
COLOR_TI_OPT = "#0072B2"     # Deep blue
COLOR_C2_OPT = "#CC79A7"     # Reddish purple
COLOR_C2_REAL = "#D55E00"    # Vermillion
COLOR_PHI = "#228B22"        # Forest green
COLOR_GUIDE = "#808080"      # Gray

LINEWIDTH = 3.2
PHI_LINEWIDTH = 3.5
GUIDE_LINEWIDTH = 2.0


# ===============================
# HELPERS
# ===============================
def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pair(data, key, i, j):
    """Extract pair-dependent data."""
    arr = np.asarray(
        data[key],
        dtype=float
    )

    if arr.ndim == 3:
        return arr[i, j]

    elif arr.ndim == 2:
        return arr

    return arr


# ===============================
# MAIN PLOTTER
# ===============================
def plot_colloid_polymer(
    delta_c_file,
    g_file,
    image_file="colloid_colloid.png",
    save_path="colloid_polymer_a_r.png"
):

    # ===============================
    # COLLOID-COLLOID PAIR
    # ===============================
    i, j = 1, 1


    # ===============================
    # LOAD DATA
    # ===============================
    data_dc = load_json(
        delta_c_file
    )

    data_g = load_json(
        g_file
    )

    r = np.asarray(
        data_dc["r"],
        dtype=float
    )


    # ===============================
    # EXTRACT a(r)
    # ===============================
    dc_real = extract_pair(
        data_dc,
        "delta_c_real_ref",
        i,
        j
    )

    dc_opt = extract_pair(
        data_dc,
        "delta_c_sigma_opt_sigma_opt",
        i,
        j
    )

    Gu_real = extract_pair(
        data_g,
        "G_u_r_real",
        i,
        j
    )

    Gu_opt = extract_pair(
        data_g,
        "G_u_r_sigma_opt",
        i,
        j
    )

    u_att = extract_pair(
        data_g,
        "u_attractive_real",
        i,
        j
    )


    # ===============================
    # INTEGRATED KERNEL
    # 4*pi*r^2*a(r)
    # ===============================
    pref = (
        4.0
        * np.pi
        * r**2
    )

    A_real = pref * Gu_real
    A_optG = pref * Gu_opt
    A_opt = pref * dc_opt
    A_ref = pref * dc_real
    A_att = pref * u_att


    # ===============================
    # FIGURE
    # ===============================
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    # =====================================================
    # MAIN PANEL
    # =====================================================

    # ===============================
    # REAL TI
    # Solid line at lowest layer
    # ===============================
    ax.plot(
        r,
        A_real,
        linewidth=LINEWIDTH,
        color=COLOR_TI_REAL,
        linestyle="-",
        alpha=1.0,
        label=r"$a_{\rm TI}^{\rm real}$",
        zorder=1
    )


    # ===============================
    # OPTIMIZED TI
    # ===============================
    ax.plot(
        r,
        A_optG,
        linewidth=LINEWIDTH,
        color=COLOR_TI_OPT,
        linestyle="--",
        alpha=1.0,
        label=r"$a_{\rm TI}^{\rm opt}$",
        zorder=3
    )


    # ===============================
    # OPTIMIZED c^(2)
    # ===============================
    ax.plot(
        r,
        A_opt,
        linewidth=LINEWIDTH,
        color=COLOR_C2_OPT,
        linestyle="-.",
        alpha=1.0,
        label=r"$a_{c^{(2)}}^{\rm opt}$",
        zorder=5
    )


    # ===============================
    # REAL c^(2)
    # ===============================
    ax.plot(
        r,
        A_ref,
        linewidth=LINEWIDTH,
        color=COLOR_C2_REAL,
        linestyle="--",
        alpha=1.0,
        label=r"$a_{c^{(2)}}^{\rm real}$",
        zorder=4
    )


    # ===============================
    # ATTRACTIVE POTENTIAL
    # ===============================
    ax.plot(
        r,
        A_att,
        linewidth=PHI_LINEWIDTH,
        color=COLOR_PHI,
        linestyle=":",
        alpha=1.0,
        label=r"$a_{\phi}$",
        zorder=3
    )


    # ===============================
    # y = 0 REFERENCE
    # GRAY DASHED
    # ===============================
    ax.axhline(
        y=0.0,
        color=COLOR_GUIDE,
        linewidth=GUIDE_LINEWIDTH,
        linestyle="--",
        alpha=0.9,
        zorder=2
    )


    # ===============================
    # MAIN AXES
    # ===============================
    ax.set_xlabel(
        r"$r$",
        fontsize=24,
        labelpad=-15
    )

    ax.set_ylabel(
        r"$4\pi r^2 a_{\rm cc}(r)$",
        fontsize=24,
        labelpad=-30
    )


    # ===============================
    # MAIN LIMITS
    # ===============================
    ax.set_xlim(
        0.0,
        5.0
    )

    # Slightly more space above y = 0
    ax.set_ylim(
        -19.0,
        4.0
    )


    # ===============================
    # MAIN TICKS
    # ===============================
    ax.set_xticks([
        0.0,
        1.5,
        3.0,
        4.5
    ])

    ax.set_yticks([
        0.0,
        -3.0,
        -15.0
    ])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True,
        width=1.6,
        length=6
    )


    # ===============================
    # MAIN LEGEND
    # ===============================
    ax.legend(
        fontsize=22,
        frameon=False,
        loc="lower right",
        handletextpad=0.3,
        handlelength=2.2
    )


    # =====================================================
    # INSET
    # =====================================================
    axins = inset_axes(
        ax,
        width="36%",
        height="60%",
        loc="upper left",
        bbox_to_anchor=(
            0.37,
            -0.1,
            1.0,
            0.60
        ),
        bbox_transform=ax.transAxes
    )


    # ===============================
    # REAL TI
    # ===============================
    axins.plot(
        r,
        Gu_real,
        color=COLOR_TI_REAL,
        linewidth=LINEWIDTH,
        linestyle="-",
        alpha=1.0,
        zorder=1
    )


    # ===============================
    # OPTIMIZED TI
    # ===============================
    axins.plot(
        r,
        Gu_opt,
        color=COLOR_TI_OPT,
        linestyle="--",
        linewidth=LINEWIDTH,
        alpha=1.0,
        zorder=3
    )


    # ===============================
    # OPTIMIZED c^(2)
    # ===============================
    axins.plot(
        r,
        dc_opt,
        color=COLOR_C2_OPT,
        linestyle="-.",
        linewidth=LINEWIDTH,
        alpha=1.0,
        zorder=5
    )


    # ===============================
    # REAL c^(2)
    # ===============================
    axins.plot(
        r,
        dc_real,
        color=COLOR_C2_REAL,
        linestyle="--",
        linewidth=LINEWIDTH,
        alpha=1.0,
        zorder=4
    )


    # ===============================
    # ATTRACTIVE POTENTIAL
    # ===============================
    axins.plot(
        r,
        u_att,
        color=COLOR_PHI,
        linestyle=":",
        linewidth=PHI_LINEWIDTH,
        alpha=1.0,
        zorder=3
    )


    # ===============================
    # y = 0 REFERENCE
    # GRAY DASHED
    # ===============================
    axins.axhline(
        y=0.0,
        color=COLOR_GUIDE,
        linewidth=GUIDE_LINEWIDTH,
        linestyle=":",
        alpha=0.9,
        zorder=2
    )


    # ===============================
    # INSET AXES
    # ===============================
    axins.set_ylabel(
        r"$a_{\rm cc}(r)$",
        fontsize=24,
        labelpad=-34
    )


    # ===============================
    # INSET LIMITS
    # ===============================
    axins.set_xlim(
        0.0,
        4.0
    )

    # Slightly more space above y = 0
    axins.set_ylim(
        -1.6,
        0.2
    )


    # ===============================
    # INSET TICKS
    # ===============================
    axins.set_xticks([
        0.0,
        2.0,
        4.0
    ])

    axins.set_yticks([
        0.0,
        -1.5
    ])

    axins.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True,
        width=1.4,
        length=6
    )


    # ===============================
    # INSET BORDER
    # ===============================
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)


    # =====================================================
    # TOP IMAGE
    # =====================================================
    img = mpimg.imread(
        image_file
    )

    ax_img = fig.add_axes(
        [
            0.50,
            0.55,
            0.17,
            0.17
        ]
    )

    ax_img.imshow(
        img
    )

    ax_img.axis(
        "off"
    )


    # ===============================
    # SAVE
    # ===============================
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=800,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {save_path}"
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":

    plot_colloid_polymer(
        "delta_c_results.json",
        "result_G_of_r.json",
        image_file="colloid_colloid.png",
        save_path="colloid_polymer_a_r.png"
    )
