import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# ===============================
# Configuration
# ===============================
json_file = "result_sigma_analysis.json"
out_dir = Path("rdf_gamma_c_all_pairs")
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
with open(json_file, "r") as f:
    data = json.load(f)

r = np.array(data["r"])
sigma_opt = np.array(data["sigma_opt"])
sigma_bh  = np.array(data["sigma_bh"])

hard_core_pairs = [tuple(map(int, k.split(","))) for k in data["bh_meta"].keys()]

# g, c, gamma dictionaries
g_target = {k: np.array(v) for k, v in data["g_real"].items()}
g_ref    = {k: np.array(v) for k, v in data["g_ref_hard"].items()}
g_opt    = {k: np.array(v) for k, v in data["g_rep_sigma_opt"].items()}
g_bh     = {k: np.array(v) for k, v in data["g_rep_sigma_bh"].items()}

c_target = {k: np.array(v) for k, v in data["c_real"].items()}
c_ref    = {k: np.array(v) for k, v in data["c_ref_hard"].items()}
c_opt    = {k: np.array(v) for k, v in data["c_rep_sigma_opt"].items()}
c_bh     = {k: np.array(v) for k, v in data["c_rep_sigma_bh"].items()}

gamma_target = {k: np.array(v) for k, v in data["gamma_real"].items()}
gamma_ref    = {k: np.array(v) for k, v in data["gamma_ref_hard"].items()}
gamma_opt    = {k: np.array(v) for k, v in data["gamma_rep_sigma_opt"].items()}
gamma_bh     = {k: np.array(v) for k, v in data["gamma_rep_sigma_bh"].items()}

example_state = next(iter(g_target))
N = g_target[example_state].shape[0]
all_pairs = [(i, j) for i in range(N) for j in range(i, N)]

# ===============================
# Plot all pairs with styled figure
# ===============================
n_pairs = len(all_pairs)
fig, axs = plt.subplots(n_pairs, 2, figsize=(8, 10 ), sharex=True)
if n_pairs == 1:
    axs = np.array([axs])  # ensure 2D indexing

# subplot letters
letters = [chr(97+i) for i in range(n_pairs*2)]  # 'a', 'b', 'c', ...

for row_idx, (i, j) in enumerate(all_pairs):
    ax_g = axs[row_idx, 0]
    ax_cg = axs[row_idx, 1]

    # -------------------------------
    # LEFT: g(r)
    # -------------------------------
    ax_g.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        zorder=0,
        label="_nolegend_"
    )

    g_real = g_target[example_state][i, j]
    g_ref_ = g_ref[example_state][i, j]
    g_opt_ = g_opt[example_state][i, j]
    g_bh_  = g_bh[example_state][i, j]

    ax_g.plot(r, g_real, lw=4, color="black")
    ax_g.plot(r, g_ref_, "--", lw=4, color="gray")
    ax_g.plot(r, g_opt_, ":", lw=4, color="blue")
    ax_g.plot(r, g_bh_, "-.", lw=4, color="red")

    ax_g.set_xlim(0.0, 4)

    
    name_i = "c" if i == 0 else "p"
    name_j = "c" if j == 0 else "p"

    ax_g.set_ylabel(rf"$g_{{{name_i}{name_j}}}(r)$", fontsize=19, labelpad=-15)
    ax_g.set_title(rf"\rm ({letters[2*row_idx]})", fontsize=18, loc="center", pad=-15)
    ax_g.tick_params(axis="y", labelsize=18)

    # >>> MIN / MAX TICKS FOR g(r)
    g_min = min(
        g_real.min(),
        g_ref_.min(),
        g_opt_.min(),
        g_bh_.min()
    )
    g_max = max(
        g_real.max(),
        g_ref_.max(),
        g_opt_.max(),
        g_bh_.max()
    )
    g_min_tick = np.round(g_min, 1)
    g_max_tick = np.round(g_max, 1)

    ax_g.set_yticks([g_min_tick, g_max_tick])


    if row_idx == 0:
        ax_g.legend(
            [r"$g_{\rm real}$", r"$g_{\rm ref}$", r"$g_{\sigma_{\rm opt}}$", r"$g_{\sigma_{\rm BH}}$"],
            fontsize=16,
            ncol=2,
            frameon=True,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.0),
            columnspacing=0.4,
            handletextpad=0.3,
            borderaxespad=0.2
        )

    # -------------------------------
    # RIGHT: c(r) and gamma(r)
    # -------------------------------
    ax_cg.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        zorder=0,
        label="_nolegend_"
    )

    c_real = c_target[example_state][i, j]
    c_ref_ = c_ref[example_state][i, j]
    c_opt_ = c_opt[example_state][i, j]
    c_bh_  = c_bh[example_state][i, j]

    gma_real = gamma_target[example_state][i, j]
    gma_ref_ = gamma_ref[example_state][i, j]
    gma_opt_ = gamma_opt[example_state][i, j]
    gma_bh_  = gamma_bh[example_state][i, j]

    ax_cg.plot(r, c_real, lw=4, color="black")
    ax_cg.plot(r, c_ref_, "--", lw=4, color="gray")
    ax_cg.plot(r, c_opt_, ":", lw=4, color="blue")
    ax_cg.plot(r, c_bh_, "-.", lw=4, color="red")

    ax_cg.plot(r, gma_real, lw=4, color="black", alpha=0.5)
    ax_cg.plot(r, gma_ref_, "--", lw=4, color="gray", alpha=0.5)
    ax_cg.plot(r, gma_opt_, ":", lw=4, color="blue", alpha=0.5)
    ax_cg.plot(r, gma_bh_, "-.", lw=4, color="red", alpha=0.5)

    ax_cg.set_xlim(0.0, 4)
    ax_cg.tick_params(axis="y", labelsize=18)

    ax_cg.set_ylabel(
        rf"$c_{{{name_i}{name_j}}}(r), \gamma_{{{name_i}{name_j}}}(r)$",
        fontsize=19,
        labelpad=-25
    )
    ax_cg.set_title(rf"\rm ({letters[2*row_idx+1]})", fontsize=18, loc="center", pad=-15)

    # >>> MIN / MAX TICKS FOR c, γ
    cg_min = min(
        c_real.min(), c_ref_.min(), c_opt_.min(), c_bh_.min(),
        gma_real.min(), gma_ref_.min(), gma_opt_.min(), gma_bh_.min()
    )
    cg_max = max(
        c_real.max(), c_ref_.max(), c_opt_.max(), c_bh_.max(),
        gma_real.max(), gma_ref_.max(), gma_opt_.max(), gma_bh_.max()
    )
    cg_min_tick = np.round(cg_min, 1)
    cg_max_tick = np.round(cg_max, 1)

    ax_cg.set_yticks([cg_min_tick, cg_max_tick])


    if row_idx == 0:
        ax_cg.legend(
            [r"$c_{\rm real}$", r"$c_{\rm ref}$", r"$c_{\sigma_{\rm opt}}$", r"$c_{\sigma_{\rm BH}}$",
             r"$\gamma_{\rm real}$", r"$\gamma_{\rm ref}$",
             r"$\gamma_{\sigma_{\rm opt}}$", r"$\gamma_{\sigma_{\rm BH}}$"],
            fontsize=16,
            ncol=2,
            frameon=True,
            loc="lower right",
            bbox_to_anchor=(1.03, -0.035),
            columnspacing=0.4,
            handletextpad=0.3
        )

# -------------------------------
# X-labels for bottom row
# -------------------------------
for ax in axs[-1, :]:
    ax.set_xlabel(r"$r$", fontsize=19)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.tick_params(axis="x", labelsize=18)

plt.tight_layout()
plt.savefig(out_dir / "rdf_c_gamma_all_pairs.png", dpi=600, bbox_inches="tight")
plt.close(fig)

print("✅ Styled g(r), c(r), gamma(r) plot saved in:", out_dir)

