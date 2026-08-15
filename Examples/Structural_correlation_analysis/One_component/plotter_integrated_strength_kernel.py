import json
import numpy as np
import matplotlib.pyplot as plt
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
# COLOR SCHEME
# ===============================
COLOR_REAL = "#000000"       # Black
COLOR_TI_OPT = "#0072B2"     # Deep blue
COLOR_C2_OPT = "#CC79A7"     # Reddish purple
COLOR_C2_REAL = "#D55E00"    # Vermillion
COLOR_PHI = "#228B22"        # Forest green
COLOR_GUIDE = "#808080"      # Gray


# ===============================
# LINE WIDTHS
# ===============================
LINEWIDTH = 3.2
PHI_LINEWIDTH = 3.5
GUIDE_LINEWIDTH = 2.0


# ===============================
# HELPERS
# ===============================
def load_json(path):
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_pair(data, key, i, j):
    """Extract pair-dependent data from a JSON array."""
    arr = np.asarray(data[key], dtype=float)

    if arr.ndim == 3:
        return arr[i, j]

    if arr.ndim == 2:
        return arr

    return arr


# ===============================
# MAIN PLOTTER
# ===============================
def plot_integrated_kernel(
    delta_c_file,
    g_file,
    i=0,
    j=0,
    save_path="a_r_plot.png"
):

    # ===============================
    # LOAD DATA
    # ===============================
    data_dc = load_json(delta_c_file)
    data_g = load_json(g_file)

    r = np.asarray(
        data_dc["r"],
        dtype=float
    )


    # ===============================
    # EXTRACT V(r)
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
    # a(r) = 4*pi*r^2*V(r)
    # ===============================
    prefactor = (
        4.0
        * np.pi
        * r**2
    )

    a_real = prefactor * Gu_real
    a_optG = prefactor * Gu_opt
    a_opt = prefactor * dc_opt
    a_ref = prefactor * dc_real
    a_att = prefactor * u_att


    # ===============================
    # FIGURE
    # ===============================
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    # ===============================
    # REAL TI
    # ===============================
    ax.plot(
        r,
        a_real,
        linewidth=LINEWIDTH + 0.3,
        color=COLOR_REAL,
        linestyle="-",
        label=r"$a_{\rm TI}^{\rm real}$",
        alpha=1.0,
        zorder=1
    )


    # ===============================
    # OPTIMIZED TI
    # ===============================
    ax.plot(
        r,
        a_optG,
        linewidth=LINEWIDTH,
        color=COLOR_TI_OPT,
        linestyle="--",
        label=r"$a_{\rm TI}^{\rm opt}$",
        alpha=1.0,
        zorder=3
    )


    # ===============================
    # OPTIMIZED c^(2)
    # ===============================
    ax.plot(
        r,
        a_opt,
        linewidth=LINEWIDTH,
        color=COLOR_C2_OPT,
        linestyle="-.",
        label=r"$a_{c^{(2)}}^{\rm opt}$",
        alpha=1.0,
        zorder=5
    )


    # ===============================
    # REAL c^(2)
    # ===============================
    ax.plot(
        r,
        a_ref,
        linewidth=LINEWIDTH,
        color=COLOR_C2_REAL,
        linestyle="--",
        label=r"$a_{c^{(2)}}^{\rm real}$",
        alpha=1.0,
        zorder=4
    )


    # ===============================
    # ATTRACTIVE POTENTIAL
    # ===============================
    ax.plot(
        r,
        a_att,
        linewidth=PHI_LINEWIDTH,
        color=COLOR_PHI,
        linestyle=":",
        label=r"$a_{\phi}$",
        alpha=1.0,
        zorder=3
    )


    # ===============================
    # y = 0 REFERENCE
    # GRAY DOTTED
    # ===============================
    ax.axhline(
        0.0,
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
        r"$4\pi r^2 a(r)$",
        fontsize=24,
        labelpad=-30
    )

    ax.set_xlim(
        0.0,
        5.0
    )

    ax.set_ylim(
        -15.0,
        1.0
    )

    ax.set_xticks([
        0.0,
        1.5,
        3.0,
        4.5
    ])

    ax.set_yticks([
        0.0,
        -3.0,
        -12.0
    ])


    # ===============================
    # MAIN TICKS
    # ===============================
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


    # ===============================
    # INSET
    # ===============================
    axins = inset_axes(
        ax,
        width="30%",
        height="60%",
        loc="upper left",
        bbox_to_anchor=(
            0.40,
            0.0,
            1.0,
            0.6
        ),
        bbox_transform=ax.transAxes
    )


    # ===============================
    # REAL TI
    # ===============================
    axins.plot(
        r,
        Gu_real,
        color=COLOR_REAL,
        linewidth=LINEWIDTH + 0.3,
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
        linewidth=LINEWIDTH,
        linestyle="--",
        alpha=1.0,
        zorder=5
    )


    # ===============================
    # OPTIMIZED c^(2)
    # ===============================
    axins.plot(
        r,
        dc_opt,
        color=COLOR_C2_OPT,
        linewidth=LINEWIDTH,
        linestyle="-.",
        alpha=1.0,
        zorder=4
    )


    # ===============================
    # REAL c^(2)
    # ===============================
    axins.plot(
        r,
        dc_real,
        color=COLOR_C2_REAL,
        linewidth=LINEWIDTH,
        linestyle="--",
        alpha=1.0,
        zorder=3
    )


    # ===============================
    # ATTRACTIVE POTENTIAL
    # ===============================
    axins.plot(
        r,
        u_att,
        color=COLOR_PHI,
        linewidth=PHI_LINEWIDTH,
        linestyle=":",
        alpha=1.0,
        zorder=2
    )


    # ===============================
    # y = 0 REFERENCE
    # GRAY DOTTED
    # ===============================
    axins.axhline(
        0.0,
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
        r"$a(r)$",
        fontsize=24,
        labelpad=-20
    )

    axins.set_xlim(
        0.0,
        4.0
    )

    axins.set_ylim(
        -1.2,
        0.2
    )

    axins.set_xticks([
        0.0,
        2.0,
        4.0
    ])

    axins.set_yticks([
        0.0,
        -1.0
    ])


    # ===============================
    # INSET TICKS
    # ===============================
    axins.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True,
        width=1.4,
        length=5
    )


    # ===============================
    # INSET BORDER
    # ===============================
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)


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
        f"Integrated-kernel plot saved to: "
        f"{save_path}"
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":

    plot_integrated_kernel(
        "delta_c_results.json",
        "result_G_of_r.json",
        i=0,
        j=0,
        save_path="a_r_plot.png"
    )
