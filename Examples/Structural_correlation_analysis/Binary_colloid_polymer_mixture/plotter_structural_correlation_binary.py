import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
# CONSISTENT COLOR SCHEME
# ===============================
COLOR_REAL = "#000000"    # Black
COLOR_WCA = "#D55E00"     # Vermillion
COLOR_OPT = "#0072B2"     # Deep blue
COLOR_BH = "#CC79A7"      # Reddish purple
COLOR_GUIDE = "#555555"   # Neutral grey


# ===============================
# LINE WIDTHS AND LAYERS
# ===============================
LINEWIDTH = 4.2
GUIDE_LINEWIDTH = 2.2

# Solid lines must remain on the bottom layer
ZORDER_SOLID = 1
ZORDER_DASHED = 2
ZORDER_DASHDOT = 3
ZORDER_DOTTED = 4


# ===============================
# DENSITIES
# ===============================
rho = {
    0: 0.6,      # polymer
    1: 0.096     # colloid
}


# ===============================
# HELPERS
# ===============================
def load_json(path):
    """Load a JSON data file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_pair(data, key, i, j):
    """Extract the requested pair-dependent array."""
    array = np.asarray(data[key], dtype=float)

    if array.ndim == 3:
        return array[i, j]

    if array.ndim == 2:
        return array

    return array


# ===============================
# STRUCTURE FACTOR
# ===============================
def structure_factor_ij(
    k,
    r,
    g_ij,
    rho_i,
    rho_j,
    i,
    j
):
    """
    Calculate the partial structure factor S_ij(k).
    """
    kr = np.outer(k, r)

    # np.sinc(x) = sin(pi*x)/(pi*x)
    # Therefore, np.sinc(kr/pi) = sin(kr)/(kr).
    # This safely handles kr = 0.
    sinc = np.sinc(kr / np.pi)

    integrand = (g_ij - 1.0) * r**2

    integral = np.trapz(
        integrand[np.newaxis, :] * sinc,
        x=r,
        axis=1
    )

    delta = 1.0 if i == j else 0.0

    return (
        delta
        + 4.0
        * np.pi
        * np.sqrt(rho_i * rho_j)
        * integral
    )


# ===============================
# COMMON CURVE PLOTTER
# ===============================
def plot_four_curves(
    ax,
    x,
    real,
    wca,
    opt,
    bh,
    include_labels=False
):
    """
    Plot four curves using the standard color, line-style,
    width, and layer hierarchy.

    The solid real-system curve is always placed underneath
    all patterned curves.
    """
    real_label = r"$\rm real$" if include_labels else None
    wca_label = (
        r"$\rm ref_{\rm WCA}$"
        if include_labels
        else None
    )
    opt_label = (
        r"$\rm ref_{\sigma_{\rm opt}}$"
        if include_labels
        else None
    )
    bh_label = (
        r"$\rm ref_{\sigma_{\rm BH}}$"
        if include_labels
        else None
    )

    # Solid real-system curve: bottom visual layer
    ax.plot(
        x,
        real,
        color=COLOR_REAL,
        linewidth=LINEWIDTH,
        linestyle="-",
        alpha=1.0,
        label=real_label,
        zorder=ZORDER_SOLID
    )

    # Dashed WCA curve
    ax.plot(
        x,
        wca,
        color=COLOR_WCA,
        linewidth=LINEWIDTH,
        linestyle="--",
        alpha=1.0,
        label=wca_label,
        zorder=ZORDER_DASHED
    )

    # Dash-dot optimized-diameter curve
    ax.plot(
        x,
        opt,
        color=COLOR_OPT,
        linewidth=LINEWIDTH,
        linestyle="-.",
        alpha=1.0,
        label=opt_label,
        zorder=ZORDER_DASHDOT
    )

    # Dotted Barker–Henderson curve
    ax.plot(
        x,
        bh,
        color=COLOR_BH,
        linewidth=LINEWIDTH,
        linestyle=":",
        alpha=1.0,
        label=bh_label,
        zorder=ZORDER_DOTTED
    )


# ===============================
# MAIN PLOT
# ===============================
def plot_three_by_three(
    sigma_file,
    save_path="three_by_three.png"
):
    data_sigma = load_json(sigma_file)

    r = np.asarray(
        data_sigma["r"],
        dtype=float
    )

    k = np.linspace(
        0.05,
        30.0,
        600
    )

    # ===============================
    # INTERACTION PAIRS
    # ===============================
    pairs = [
        ("polymer-polymer", (0, 0)),
        ("colloid-polymer", (1, 0)),
        ("colloid-colloid", (1, 1))
    ]

    # ===============================
    # FIGURE
    # Three rows: observables
    # Three columns: interaction pairs
    # ===============================
    fig, axs = plt.subplots(
        3,
        3,
        figsize=(16, 9)
    )

    for col, (label, (i, j)) in enumerate(pairs):
        rho_i = rho[i]
        rho_j = rho[j]

        # ===============================
        # EXTRACT DATA
        # ===============================
        g_real = extract_pair(
            data_sigma,
            "g_real",
            i,
            j
        )

        g_ref = extract_pair(
            data_sigma,
            "g_ref_hard",
            i,
            j
        )

        g_opt = extract_pair(
            data_sigma,
            "g_rep_sigma_opt",
            i,
            j
        )

        g_bh = extract_pair(
            data_sigma,
            "g_rep_sigma_bh",
            i,
            j
        )

        c_real = extract_pair(
            data_sigma,
            "c_real",
            i,
            j
        )

        c_ref = extract_pair(
            data_sigma,
            "c_ref_hard",
            i,
            j
        )

        c_opt = extract_pair(
            data_sigma,
            "c_rep_sigma_opt",
            i,
            j
        )

        c_bh = extract_pair(
            data_sigma,
            "c_rep_sigma_bh",
            i,
            j
        )

        # ===============================
        # STRUCTURE FACTORS
        # ===============================
        S_real = structure_factor_ij(
            k,
            r,
            g_real,
            rho_i,
            rho_j,
            i,
            j
        )

        S_ref = structure_factor_ij(
            k,
            r,
            g_ref,
            rho_i,
            rho_j,
            i,
            j
        )

        S_opt = structure_factor_ij(
            k,
            r,
            g_opt,
            rho_i,
            rho_j,
            i,
            j
        )

        S_bh = structure_factor_ij(
            k,
            r,
            g_bh,
            rho_i,
            rho_j,
            i,
            j
        )

        # =====================================================
        # ROW 1: RADIAL DISTRIBUTION FUNCTION g(r)
        # =====================================================
        ax = axs[0, col]

        ax.axhline(
            1.0,
            color=COLOR_GUIDE,
            linestyle="--",
            linewidth=GUIDE_LINEWIDTH,
            alpha=0.75,
            zorder=0
        )

        plot_four_curves(
            ax,
            r,
            g_real,
            g_ref,
            g_opt,
            g_bh,
            include_labels=(col == 0)
        )

        ax.set_xlim(
            0.0,
            3.5
        )

        ax.set_xticks([
            0.0,
            1.0,
            3.0
        ])

        if col == 0:
            ax.text(
                0.02,
                0.94,
                r"$\rm (a)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$g_{\rm pp}(r)$",
                fontsize=24,
                labelpad=-20
            )

            ax.set_yticks([
                0.5,
                1.0
            ])

            ax.legend(
                fontsize=20,
                ncol=1,
                frameon=False,
                loc="lower right",
                handlelength=2.3,
                handletextpad=0.5
            )

        elif col == 1:
            ax.text(
                0.02,
                0.98,
                r"$\rm (b)$",
                transform=ax.transAxes,
                fontsize=22,
                va="top"
            )

            ax.set_ylabel(
                r"$g_{\rm cp}(r)$",
                fontsize=24,
                labelpad=-8
            )

            ax.set_yticks([
                0.0,
                2.0
            ])

        else:
            ax.text(
                0.02,
                0.98,
                r"$\rm (c)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$g_{\rm cc}(r)$",
                fontsize=24,
                labelpad=-8
            )

            ax.set_yticks([
                0.0,
                3.0
            ])

        # =====================================================
        # ROW 2: STRUCTURE FACTOR S(k)
        # =====================================================
        ax = axs[1, col]

        if col == 1:
            reference_level = 0.0
        else:
            reference_level = 1.0

        ax.axhline(
            reference_level,
            color=COLOR_GUIDE,
            linestyle="--",
            linewidth=GUIDE_LINEWIDTH,
            alpha=0.75,
            zorder=0
        )

        plot_four_curves(
            ax,
            k,
            S_real,
            S_ref,
            S_opt,
            S_bh
        )

        ax.set_xlim(
            0.0,
            20.0
        )

        ax.set_xticks([
            0.0,
            7.0,
            14.0
        ])

        if col == 0:
            ax.text(
                0.02,
                0.94,
                r"$\rm (d)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$S_{\rm pp}(k)$",
                fontsize=24,
                labelpad=-7
            )

            ax.set_yticks([
                0.0,
                1.0
            ])

        elif col == 1:
            ax.text(
                0.02,
                0.99,
                r"$\rm (e)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$S_{\rm cp}(k)$",
                fontsize=24,
                labelpad=-25
            )

            ax.set_yticks([
                -1.0,
                0.0
            ])

        else:
            ax.text(
                0.05,
                0.98,
                r"$\rm (f)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$S_{\rm cc}(k)$",
                fontsize=24,
                labelpad=-22
            )

            ax.set_yticks([
                0.4,
                1.2
            ])

        # =====================================================
        # ROW 3: DIRECT CORRELATION FUNCTION c(r)
        # =====================================================
        ax = axs[2, col]

        ax.axhline(
            0.0,
            color=COLOR_GUIDE,
            linestyle="--",
            linewidth=GUIDE_LINEWIDTH,
            alpha=0.75,
            zorder=0
        )

        plot_four_curves(
            ax,
            r,
            c_real,
            c_ref,
            c_opt,
            c_bh
        )

        ax.set_xlim(
            0.0,
            3.5
        )

        ax.set_xticks([
            0.0,
            1.0,
            3.0
        ])

        if col == 0:
            ax.text(
                0.02,
                0.92,
                r"$\rm (g)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$c_{\rm pp}(r)$",
                fontsize=24,
                labelpad=-20
            )

            ax.set_yticks([
                0.0,
                -2.0
            ])

        elif col == 1:
            ax.text(
                0.02,
                0.92,
                r"$\rm (h)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$c_{\rm cp}(r)$",
                fontsize=24,
                labelpad=-20
            )

            ax.set_yticks([
                0.0,
                -3.0
            ])

        else:
            ax.text(
                0.02,
                0.84,
                r"$\rm (i)$",
                transform=ax.transAxes,
                fontsize=24,
                va="top"
            )

            ax.set_ylabel(
                r"$c_{\rm cc}(r)$",
                fontsize=24,
                labelpad=-20
            )

            ax.set_yticks([
                0.0,
                -6.0
            ])

    # ===============================
    # COMMON AXIS FORMATTING
    # ===============================
    for ax in axs.flat:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=22,
            direction="in",
            top=True,
            right=True
        )

    # Correct independent-variable labels:
    # row 0: g(r)
    # row 1: S(k)
    # row 2: c(r)
    for col in range(3):
        axs[0, col].set_xlabel(
            r"$r$",
            fontsize=24,
            labelpad=-20
        )

        axs[1, col].set_xlabel(
            r"$k$",
            fontsize=24,
            labelpad=-20
        )

        axs[2, col].set_xlabel(
            r"$r$",
            fontsize=24,
            labelpad=-20
        )

    plt.tight_layout()

    plt.subplots_adjust(
        hspace=0.3,
        wspace=0.35
    )

    # ===============================
    # COLUMN IMAGES
    # ===============================
    image_files = [
        "polymer_polymer.png",
        "polymer_colloid.png",
        "colloid_colloid.png"
    ]

    for col, image_file in enumerate(image_files):
        # Position based on top-row axes
        position = axs[0, col].get_position()

        image_width = 0.15
        image_height = 0.15

        x_image = (
            0.5 * (position.x0 + position.x1)
            - 0.5 * image_width
            - 0.01
        )

        y_image = position.y1

        image_axis = fig.add_axes([
            x_image,
            y_image,
            image_width,
            image_height
        ])

        image = mpimg.imread(image_file)

        image_axis.imshow(image)
        image_axis.axis("off")

    # ===============================
    # VERTICAL SEPARATORS
    # ===============================
    position_00 = axs[0, 0].get_position()
    position_01 = axs[0, 1].get_position()
    position_02 = axs[0, 2].get_position()

    # Midpoints between columns
    separator_x1 = (
        0.5 * (position_00.x1 + position_01.x0)
        - 0.03
    )

    separator_x2 = (
        0.5 * (position_01.x1 + position_02.x0)
        - 0.03
    )

    # Vertical span covering all three rows
    separator_y0 = (
        axs[2, 0].get_position().y0
        - 0.04
    )

    separator_y1 = (
        axs[0, 0].get_position().y1
        + 0.07
    )

    for separator_x in [
        separator_x1,
        separator_x2
    ]:
        separator = plt.Line2D(
            [separator_x, separator_x],
            [separator_y0, separator_y1],
            transform=fig.transFigure,
            color=COLOR_GUIDE,
            linestyle=":",
            linewidth=GUIDE_LINEWIDTH,
            alpha=0.7,
            zorder=0
        )

        fig.add_artist(separator)

    # ===============================
    # SAVE
    # ===============================
    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved reorganized 3x3 figure to: {save_path}"
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    plot_three_by_three(
        "result_sigma_analysis.json",
        save_path="three_by_three.png"
    )
