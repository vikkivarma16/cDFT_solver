import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D


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
COLOR_SIM = "#000000"         # Black
COLOR_B2 = "#228B22"          # Forest green
COLOR_C2 = "#D55E00"          # Vermillion
COLOR_TI = "#0072B2"          # Deep blue
COLOR_HIGHLIGHT = "cyan"      # Input-state highlight

# Critical-point star
COLOR_STAR_FACE = "#FFD700"    # Yellow/gold filling
COLOR_STAR_EDGE = "#000000"    # Black border


# ===============================
# LINE AND MARKER SETTINGS
# ===============================
LINEWIDTH = 3.2
DATA_MARKER_SIZE = 8
STAR_MARKER_SIZE = 15
STAR_EDGE_WIDTH = 2.2
HIGHLIGHT_SIZE = 120


# =========================================================
# HIGHLIGHT INPUT STATE POINT
# =========================================================
def highlight_density_point(
    ax,
    rho,
    temperature,
    rho_target=0.05
):
    """
    Highlight the simulation state point nearest rho_target.
    """
    index = np.argmin(
        np.abs(rho - rho_target)
    )

    ax.scatter(
        rho[index],
        temperature[index],
        s=HIGHLIGHT_SIZE,
        marker="o",
        facecolors="none",
        edgecolors=COLOR_HIGHLIGHT,
        linewidths=3.5,
        zorder=30
    )


# =========================================================
# LOAD BINODAL DATA
# =========================================================
def load_binodal(filename):
    """
    Load inverse-temperature and coexistence-density data.

    Expected columns:
        column 0: beta = 1/T
        column 1: liquid density
        column 2: vapor density
    """
    data = np.loadtxt(filename)

    temperature = 1.0 / data[:, 0]
    rho_liquid = data[:, 1]
    rho_vapor = data[:, 2]

    order = np.argsort(temperature)

    return (
        temperature[order],
        rho_vapor[order],
        rho_liquid[order]
    )


# =========================================================
# FIT COEXISTENCE BRANCH
# =========================================================
def fit_branch(
    temperature,
    rho,
    critical_temperature,
    critical_density,
    branch="liquid"
):
    """
    Fit the five points nearest the critical temperature
    using a power-law coexistence form.
    """
    index = np.argsort(temperature)[-5:]

    temperature_fit = temperature[index]
    rho_fit = rho[index]

    def model(temperature_value, amplitude, exponent):
        argument = np.maximum(
            critical_temperature - temperature_value,
            0.0
        )

        if branch == "liquid":
            return (
                critical_density
                + amplitude * argument**exponent
            )

        return (
            critical_density
            - amplitude * argument**exponent
        )

    optimal_parameters, _ = curve_fit(
        model,
        temperature_fit,
        rho_fit,
        p0=[0.5, 0.32275],
        bounds=(
            [0.0, 0.1],
            [10.0, 0.6]
        ),
        maxfev=10000
    )

    amplitude, exponent = optimal_parameters

    print(
        f"{branch}: "
        f"A = {amplitude:.4f}, "
        f"beta = {exponent:.4f}"
    )

    temperature_smooth = np.linspace(
        np.min(temperature),
        critical_temperature,
        400
    )

    rho_smooth = model(
        temperature_smooth,
        amplitude,
        exponent
    )

    return temperature_smooth, rho_smooth


# =========================================================
# LOAD DATA
# =========================================================
T_b, rv_b, rl_b = load_binodal(
    "binodal_delta_b.txt"
)

T_c, rv_c, rl_c = load_binodal(
    "binodal_simulation.txt"
)

T_dc, rv_dc, rl_dc = load_binodal(
    "binodal_delta_c.txt"
)

T_rdf, rv_rdf, rl_rdf = load_binodal(
    "binodal_rdf.txt"
)


# =========================================================
# CRITICAL POINTS
# =========================================================
Tc_b, rhoc_b = 1.26, 0.26
Tc_c, rhoc_c = 1.31, 0.30
Tc_dc, rhoc_dc = 1.26, 0.26
Tc_rdf, rhoc_rdf = 1.26, 0.26


print("\nCritical points:")

print(
    f"Simulation: "
    f"rho_c = {rhoc_c:.3f}, "
    f"T_c = {Tc_c:.3f}"
)

print(
    f"A_B2: "
    f"rho_c = {rhoc_b:.3f}, "
    f"T_c = {Tc_b:.3f}"
)

print(
    f"A_c2: "
    f"rho_c = {rhoc_dc:.3f}, "
    f"T_c = {Tc_dc:.3f}"
)

print(
    f"A_TI: "
    f"rho_c = {rhoc_rdf:.3f}, "
    f"T_c = {Tc_rdf:.3f}"
)


# =========================================================
# FIT LIQUID AND VAPOR BRANCHES
# =========================================================

# B2-based result
Tsb_l, rlb_s = fit_branch(
    T_b,
    rl_b,
    Tc_b,
    rhoc_b,
    branch="liquid"
)

Tsb_v, rvb_s = fit_branch(
    T_b,
    rv_b,
    Tc_b,
    rhoc_b,
    branch="vapor"
)


# Simulation
Tsc_l, rlc_s = fit_branch(
    T_c,
    rl_c,
    Tc_c,
    rhoc_c,
    branch="liquid"
)

Tsc_v, rvc_s = fit_branch(
    T_c,
    rv_c,
    Tc_c,
    rhoc_c,
    branch="vapor"
)


# Direct-correlation result
Tsd_l, rld_s = fit_branch(
    T_dc,
    rl_dc,
    Tc_dc,
    rhoc_dc,
    branch="liquid"
)

Tsd_v, rvd_s = fit_branch(
    T_dc,
    rv_dc,
    Tc_dc,
    rhoc_dc,
    branch="vapor"
)


# Thermodynamic-integration result
Tsr_l, rlr_s = fit_branch(
    T_rdf,
    rl_rdf,
    Tc_rdf,
    rhoc_rdf,
    branch="liquid"
)

Tsr_v, rvr_s = fit_branch(
    T_rdf,
    rv_rdf,
    Tc_rdf,
    rhoc_rdf,
    branch="vapor"
)


# =========================================================
# FIGURE
# =========================================================
fig, ax = plt.subplots(
    figsize=(8.5, 5.0)
)


# =========================================================
# FITTED CURVES
# Solid curve is always placed on the bottom visual layer
# =========================================================

# Simulation: solid black line, bottom layer
ax.plot(
    rvc_s,
    Tsc_v,
    color=COLOR_SIM,
    linewidth=LINEWIDTH,
    linestyle="-",
    alpha=1.0,
    zorder=1
)

ax.plot(
    rlc_s,
    Tsc_l,
    color=COLOR_SIM,
    linewidth=LINEWIDTH,
    linestyle="-",
    alpha=1.0,
    zorder=1
)


# B2: dashed forest-green line
ax.plot(
    rvb_s,
    Tsb_v,
    color=COLOR_B2,
    linewidth=LINEWIDTH,
    linestyle="--",
    alpha=1.0,
    zorder=2
)

ax.plot(
    rlb_s,
    Tsb_l,
    color=COLOR_B2,
    linewidth=LINEWIDTH,
    linestyle="--",
    alpha=1.0,
    zorder=2
)


# c^(2): dash-dot vermillion line
ax.plot(
    rvd_s,
    Tsd_v,
    color=COLOR_C2,
    linewidth=LINEWIDTH,
    linestyle="-.",
    alpha=1.0,
    zorder=3
)

ax.plot(
    rld_s,
    Tsd_l,
    color=COLOR_C2,
    linewidth=LINEWIDTH,
    linestyle="-.",
    alpha=1.0,
    zorder=3
)


# TI: dotted deep-blue line
ax.plot(
    rvr_s,
    Tsr_v,
    color=COLOR_TI,
    linewidth=LINEWIDTH,
    linestyle=":",
    alpha=1.0,
    zorder=4
)

ax.plot(
    rlr_s,
    Tsr_l,
    color=COLOR_TI,
    linewidth=LINEWIDTH,
    linestyle=":",
    alpha=1.0,
    zorder=4
)


# =========================================================
# DATA MARKERS
# =========================================================

# Simulation
ax.plot(
    rv_c,
    T_c,
    linestyle="none",
    marker="^",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor=COLOR_SIM,
    markeredgewidth=2.5,
    label=r"$\rm Sim$",
    zorder=10
)

ax.plot(
    rl_c,
    T_c,
    linestyle="none",
    marker="^",
    markersize=9,
    markerfacecolor="none",
    markeredgecolor=COLOR_SIM,
    markeredgewidth=2.5,
    zorder=10
)


# B2-based result
ax.plot(
    rv_b,
    T_b,
    linestyle="none",
    marker="d",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_B2,
    markeredgewidth=2.5,
    label=r"$A_{B_2}$",
    zorder=11
)

ax.plot(
    rl_b,
    T_b,
    linestyle="none",
    marker="d",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_B2,
    markeredgewidth=2.5,
    zorder=11
)


# Direct-correlation result
ax.plot(
    rv_dc,
    T_dc,
    linestyle="none",
    marker="o",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_C2,
    markeredgewidth=2.5,
    label=r"$A_{c^{(2)}}$",
    zorder=12
)

ax.plot(
    rl_dc,
    T_dc,
    linestyle="none",
    marker="o",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_C2,
    markeredgewidth=2.5,
    zorder=12
)


# Thermodynamic-integration result
ax.plot(
    rv_rdf,
    T_rdf,
    linestyle="none",
    marker="s",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_TI,
    markeredgewidth=2.5,
    label=r"$A_{\rm TI}$",
    zorder=13
)

ax.plot(
    rl_rdf,
    T_rdf,
    linestyle="none",
    marker="s",
    markersize=DATA_MARKER_SIZE,
    markerfacecolor="none",
    markeredgecolor=COLOR_TI,
    markeredgewidth=2.5,
    zorder=13
)


# =========================================================
# HIGHLIGHT SIMULATION INPUT STATE POINT
# =========================================================
highlight_density_point(
    ax,
    rv_c,
    T_c,
    rho_target=0.05
)


# =========================================================
# CRITICAL-POINT STARS
# Black border with yellow filling
# =========================================================
def plot_critical_star(
    ax,
    critical_density,
    critical_temperature,
    zorder,
    label=None
):
    """
    Plot a yellow critical-point star with a black border.

    The label is supplied only once to prevent repeated
    critical-point entries in the legend.
    """
    ax.plot(
        critical_density,
        critical_temperature,
        linestyle="none",
        marker="*",
        markersize=STAR_MARKER_SIZE,
        markerfacecolor=COLOR_STAR_FACE,
        markeredgecolor=COLOR_STAR_EDGE,
        markeredgewidth=STAR_EDGE_WIDTH,
        label=label,
        zorder=zorder
    )


# Simulation critical point
# This star supplies the critical-point legend entry
plot_critical_star(
    ax,
    rhoc_c,
    Tc_c,
    zorder=21,
    label=r"$\rm Critical\ point$"
)


# B2 critical point
plot_critical_star(
    ax,
    rhoc_b,
    Tc_b,
    zorder=22
)


# c^(2) critical point
plot_critical_star(
    ax,
    rhoc_dc,
    Tc_dc,
    zorder=23
)


# TI critical point
plot_critical_star(
    ax,
    rhoc_rdf,
    Tc_rdf,
    zorder=24
)


# =========================================================
# AXES AND LABELS
# =========================================================
ax.set_xlim(
    0.0,
    0.75
)

ax.set_xlabel(
    r"$\rho$",
    fontsize=24
)

ax.set_ylabel(
    r"$k_{\rm B}T/\epsilon$",
    fontsize=24
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=24,
    direction="in",
    top=True,
    right=True
)


# =========================================================
# MAIN LEGEND
# =========================================================
main_legend = ax.legend(
    frameon=False,
    fontsize=20,
    loc="lower left",
    bbox_to_anchor=(0.3, 0.2),
    handlelength=2.0,
    handletextpad=0.1,
    labelspacing=0.4,
    borderaxespad=0.0
)

# Preserve main legend before adding the second legend
ax.add_artist(main_legend)


# =========================================================
# SEPARATE INPUT-STATE LEGEND
# Cyan hollow circle
# =========================================================
input_state_handle = Line2D(
    [0],
    [0],
    linestyle="none",
    marker="o",
    markersize=10,
    markerfacecolor="none",
    markeredgecolor=COLOR_HIGHLIGHT,
    markeredgewidth=3.5,
    label=r"$\rm Input\ state\ point$"
)

ax.legend(
    handles=[input_state_handle],
    frameon=False,
    fontsize=20,
    loc="lower left",
    bbox_to_anchor=(0.315, 0.08),
    handlelength=1.4,
    handletextpad=0.3,
    borderaxespad=0.0
)


# =========================================================
# SAVE
# =========================================================
plt.tight_layout()

plt.savefig(
    "binodal_all_comparison.png",
    dpi=800,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved: binodal_all_comparison.png")
