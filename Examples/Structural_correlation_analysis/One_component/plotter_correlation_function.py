import json
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# GLOBAL STYLE
# ===============================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble":
        r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}",

    # Thicker axes and ticks for publication-quality figures
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

# ===============================
# PUBLICATION-FRIENDLY COLORS
# Okabe-Ito inspired palette
# ===============================
COLOR_REAL = "#000000"       # Black
COLOR_WCA = "#D55E00"        # Vermillion
COLOR_OPT = "#0072B2"        # Deep blue
COLOR_BH = "#CC79A7"  # Reddish purple
COLOR_GUIDE = "#555555"      # Neutral grey

# Main and auxiliary line widths
LINEWIDTH = 3.2
GUIDE_LINEWIDTH = 2.3

# ===============================
# DENSITIES
# ===============================
rho = {
    0: 0.4
}

# ===============================
# HELPERS
# ===============================
def load_json(path):
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_pair(data, key, i, j):
    """Extract the requested pair correlation data."""
    arr = np.asarray(data[key], dtype=float)

    if arr.ndim == 3:
        return arr[i, j]

    if arr.ndim == 2:
        return arr

    return arr


# ===============================
# PARTIAL STRUCTURE FACTOR
# ===============================
def structure_factor_ij(k, r, g_ij, rho_i, rho_j, i, j):
    """
    Calculate the partial structure factor S_ij(k).
    """
    kr = np.outer(k, r)

    # np.sinc(x) = sin(pi*x)/(pi*x), so np.sinc(kr/pi)
    # gives sin(kr)/(kr) while safely handling kr = 0.
    sinc = np.sinc(kr / np.pi)

    integrand = (g_ij - 1.0) * r**2
    integral = np.trapz(
        integrand[np.newaxis, :] * sinc,
        r,
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
# MAIN PLOTTER
# ===============================
def plot_full_analysis(
    sigma_file,
    delta_c_file,
    g_file,
    i=0,
    j=0,
    save_path="analysis_3x1.png"
):
    data_sigma = load_json(sigma_file)
    data_dc = load_json(delta_c_file)
    data_g = load_json(g_file)

    r = np.asarray(data_sigma["r"], dtype=float)

    # ===============================
    # SIGMA-ANALYSIS DATA
    # ===============================
    g_real = extract_pair(data_sigma, "g_real", i, j)
    g_ref = extract_pair(data_sigma, "g_ref_hard", i, j)
    g_opt = extract_pair(data_sigma, "g_rep_sigma_opt", i, j)
    g_bh = extract_pair(data_sigma, "g_rep_sigma_bh", i, j)

    c_real = extract_pair(data_sigma, "c_real", i, j)
    c_ref = extract_pair(data_sigma, "c_ref_hard", i, j)
    c_opt = extract_pair(data_sigma, "c_rep_sigma_opt", i, j)
    c_bh = extract_pair(data_sigma, "c_rep_sigma_bh", i, j)

    gamma_real = extract_pair(data_sigma, "gamma_real", i, j)
    gamma_ref = extract_pair(data_sigma, "gamma_ref_hard", i, j)
    gamma_opt = extract_pair(
        data_sigma,
        "gamma_rep_sigma_opt",
        i,
        j
    )
    gamma_bh = extract_pair(
        data_sigma,
        "gamma_rep_sigma_bh",
        i,
        j
    )

    u_real = extract_pair(data_sigma, "u_real", i, j)

    # ===============================
    # DELTA-c DATA
    # Retained for the same analysis structure
    # ===============================
    dc_real_ref = extract_pair(
        data_dc,
        "delta_c_real_ref",
        i,
        j
    )
    dc_real_opt = extract_pair(
        data_dc,
        "delta_c_real_sigma_opt",
        i,
        j
    )
    dc_opt_opt = extract_pair(
        data_dc,
        "delta_c_sigma_opt_sigma_opt",
        i,
        j
    )

    # ===============================
    # G(r) DATA
    # Retained for the same analysis structure
    # ===============================
    G_real = extract_pair(data_g, "G_r_real", i, j)
    G_opt = extract_pair(data_g, "G_r_sigma_opt", i, j)
    Gu_real = extract_pair(data_g, "G_u_r_real", i, j)
    Gu_opt = extract_pair(data_g, "G_u_r_sigma_opt", i, j)
    u_att = extract_pair(data_g, "u_attractive_real", i, j)

    # ===============================
    # STRUCTURE FACTORS
    # ===============================
    k = np.linspace(0.05, 30.0, 600)

    rho_i = rho[i]
    rho_j = rho[j]

    S_real = structure_factor_ij(
        k, r, g_real, rho_i, rho_j, i, j
    )
    S_ref = structure_factor_ij(
        k, r, g_ref, rho_i, rho_j, i, j
    )
    S_opt = structure_factor_ij(
        k, r, g_opt, rho_i, rho_j, i, j
    )
    S_bh = structure_factor_ij(
        k, r, g_bh, rho_i, rho_j, i, j
    )

    # ===============================
    # FIGURE: THREE PANELS
    # ===============================
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(8, 8),
        sharex=False
    )

    plt.subplots_adjust(hspace=0.08)

    # ===============================
    # PANEL (a): RDF
    # ===============================
    ax = axs[0]

    ax.axhline(
        1.0,
        color=COLOR_GUIDE,
        linestyle="--",
        linewidth=GUIDE_LINEWIDTH,
        alpha=0.75,
        zorder=1
    )

    ax.plot(
        r,
        g_opt,
        linewidth=LINEWIDTH,
        color=COLOR_OPT,
        linestyle="-.",
        label=r"$g_{\sigma_{\rm opt}}$",
        zorder=3
    )

    ax.plot(
        r,
        g_bh,
        linewidth=LINEWIDTH,
        color=COLOR_BH,
        linestyle=":",
        label=r"$g_{\sigma_{\rm BH}}$",
        zorder=3
    )

    ax.plot(
        r,
        g_real,
        linewidth=LINEWIDTH + 0.3,
        color=COLOR_REAL,
        linestyle="-",
        label=r"$g_{\rm real}$",
        zorder=1
    )

    ax.plot(
        r,
        g_ref,
        linewidth=LINEWIDTH,
        color=COLOR_WCA,
        linestyle="--",
        label=r"$g_{\rm ref}$",
        zorder=4
    )

    ax.set_xlabel(
        r"$r$",
        fontsize=24,
        labelpad=-20
    )
    ax.set_ylabel(
        r"$g(r)$",
        fontsize=24,
        labelpad=-20
    )

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_yticks([0.0, 1.5])
    ax.set_xlim(0.0, 3.5)

    ax.text(
        0.03,
        0.85,
        r"$\rm (a)$",
        transform=ax.transAxes,
        fontsize=24,
        va="top"
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True
    )

    ax.legend(
        fontsize=24,
        ncol=2,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.03, -0.12),
        columnspacing=0.5,
        handlelength=2.1,
        handletextpad=0.5
    )

    # ===============================
    # PANEL (b): STRUCTURE FACTOR
    # ===============================
    ax = axs[1]

    ax.axhline(
        1.0,
        color=COLOR_GUIDE,
        linestyle="--",
        linewidth=GUIDE_LINEWIDTH,
        alpha=0.75,
        zorder=1
    )

    ax.plot(
        k,
        S_opt,
        linewidth=LINEWIDTH,
        color=COLOR_OPT,
        linestyle="-.",
        label=r"$\rm ref_{\sigma_{\rm opt}}$",
        zorder=3
    )

    ax.plot(
        k,
        S_bh,
        linewidth=LINEWIDTH,
        color=COLOR_BH,
        linestyle=":",
        label=r"$\rm ref_{\sigma_{\rm BH}}$",
        zorder=3
    )

    ax.plot(
        k,
        S_real,
        linewidth=LINEWIDTH + 0.3,
        color=COLOR_REAL,
        linestyle="-",
        label=r"${\rm real}$",
        zorder=1
    )

    ax.plot(
        k,
        S_ref,
        linewidth=LINEWIDTH,
        color=COLOR_WCA,
        linestyle="--",
        label=r"${\rm ref_{\rm WCA}}$",
        zorder=4
    )

    ax.set_ylabel(
        r"$S(k)$",
        fontsize=24,
        labelpad=-35
    )
    ax.set_xlabel(
        r"$k$",
        fontsize=24,
        labelpad=-15
    )

    ax.text(
        0.03,
        0.96,
        r"$\rm (b)$",
        transform=ax.transAxes,
        fontsize=24,
        va="top"
    )

    ax.set_yticks([-0.2, 1.0])
    ax.set_xticks([0.0, 9.0, 18.0])
    ax.set_xlim(0.0, 20.0)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True
    )

    # ===============================
    # PANEL (c): DIRECT CORRELATION
    # ===============================
    ax = axs[2]

    ax.axhline(
        0.0,
        color=COLOR_GUIDE,
        linestyle="--",
        linewidth=GUIDE_LINEWIDTH,
        alpha=0.75,
        zorder=1
    )

    ax.plot(
        r,
        c_opt,
        linewidth=LINEWIDTH,
        color=COLOR_OPT,
        linestyle="-.",
        label=r"$c_{\sigma_{\rm opt}}$",
        zorder=3
    )

    ax.plot(
        r,
        c_bh,
        linewidth=LINEWIDTH,
        color=COLOR_BH,
        linestyle=":",
        label=r"$c_{\sigma_{\rm BH}}$",
        zorder=3
    )

    ax.plot(
        r,
        c_real,
        linewidth=LINEWIDTH + 0.3,
        color=COLOR_REAL,
        linestyle="-",
        label=r"$c_{\rm real}$",
        zorder=1
    )

    ax.plot(
        r,
        c_ref,
        linewidth=LINEWIDTH,
        color=COLOR_WCA,
        linestyle="--",
        label=r"$c_{\rm ref}$",
        zorder=4
    )

    ax.set_xlabel(
        r"$r$",
        fontsize=24,
        labelpad=-20
    )
    ax.set_ylabel(
        r"$c^{(2)}(r)$",
        fontsize=24,
        labelpad=-20
    )

    ax.text(
        0.03,
        0.82,
        r"$\rm (c)$",
        transform=ax.transAxes,
        fontsize=24,
        va="top"
    )

    ax.set_yticks([-4.0, 1.0])
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlim(0.0, 3.5)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        direction="in",
        top=True,
        right=True
    )

    # ===============================
    # FINALIZE
    # ===============================
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=800,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Full analysis saved to: {save_path}")


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    plot_full_analysis(
        "result_sigma_analysis.json",
        "delta_c_results.json",
        "result_G_of_r.json",
        i=0,
        j=0,
        save_path="analysis_3x1.png"
    )
