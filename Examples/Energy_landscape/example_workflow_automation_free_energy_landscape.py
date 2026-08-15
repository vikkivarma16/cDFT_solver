from cdft_solver.utils import ExecutionContext, create_unique_scratch_dir
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
from tqdm import tqdm

from cdft_solver.generators.potential_splitter.hc import hard_core_potentials 
from cdft_solver.generators.potential_splitter.mf import meanfield_potentials 
from cdft_solver.generators.potential_splitter.total import total_potentials
from cdft_solver.generators.potential_splitter.raw import raw_potentials
from cdft_solver.calculators.total_free_energy.total_free_energy import total_free_energy
from cdft_solver.calculators.total_free_energy.free_energy_exporter  import free_energy_exporter

from cdft_solver.calculators.phase_finder.thermodynamic_potentials_isocore import evaluate_canonical_state
from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isocore import coexistence_densities_isocore





# define your directory to export the data and the plots



scratch = create_unique_scratch_dir()
plots = scratch / "plots"
plots.mkdir(exist_ok=True)




# export different dictionaries bases on their functions.
ctx = ExecutionContext(
    input_file="example_input_isocore.in",
    scratch_dir=scratch,
    plots_dir=plots,
)

system =  super_dictionary_creator (ctx, export_json =  True, filename  = "input_system.json", super_key_name =  "system")




#exit (0)

hc_data = hard_core_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_hc.json",
    export_files=True
)


mf_data = meanfield_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_mf.json",
    export_files=True
)


raw_data = raw_potentials(
    ctx=ctx,
    input_data=system,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_raw.json",
    export_files=True
)




total_data = total_potentials(
    ctx=ctx,
    hc_source= hc_data,
    mf_source= mf_data,
    file_name_prefix="supplied_data_potential_total.json",
    export_files=True,
   
)

filenames = {}
filenames["hard_core"] = "supplied_data_free_energy_hard_core.json"
filenames["mean_field"] = "supplied_data_free_energy_mean_field.json"
filenames["ideal"] =  "supplied_data_free_energy_ideal.json"
filenames["hybrid"] =  "supplied_data_free_energy_hybrid.json"


free_energy  = total_free_energy(
    ctx=ctx,
    hc_data=hc_data,
    system_config=system,
    export_json=True,
    filenames = filenames
)


free_energy_exporter(
    ctx = ctx,
    total_fe =  free_energy,
    filename =  "supplied_data_free_energy_total.json",
    system_config=None,
    indent=4,
)




grid =  {}
grid ["r_max"] = 5
grid["n_points"] = 500
grid["r_min"] = 5/500  

densities  = [0.3, 0.001, 0.1]


print ("\n\n\nParticle's sizes are given as:", hc_data["sigma"], "\n\n\n")




value = coexistence_densities_isocore(
    ctx = ctx,
    config_dict=system,
    fe_res = free_energy,
    supplied_data = None,
    max_outer_iters=10,
    tol_outer=1e-3,
    tol_solver=1e-8,
    verbose=True
)





print (value)



# ============================================================
# TWO-PHASE FREE-ENERGY LANDSCAPE
# ============================================================
# ============================================================
# TWO-PHASE FREE-ENERGY LANDSCAPE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# ============================================================
# Extract coexistence information
# ============================================================

phase_1 = value["rhos_per_phase"][0]
phase_2 = value["rhos_per_phase"][1]

rho_liquid = phase_1
rho_gas    = phase_2

p = value["fractions"][0]

# ============================================================
# shorthand variables
# ============================================================

x_c_1 = rho_liquid[0]
x_c_2 = rho_gas[0]

z_c_1 = rho_liquid[2]
z_c_2 = rho_gas[2]

# ============================================================
# numerical free-energy evaluator
# ============================================================


def energy_free(rho_a, rho_b, rho_c):

    # --------------------------------------------------------
    # Update densities inside system dictionary
    # --------------------------------------------------------

    system["system"]["solution"][
        "extrinsic_constraints"
    ]["density"]["a"] = float(rho_a)

    system["system"]["solution"][
        "extrinsic_constraints"
    ]["density"]["b"] = float(rho_b)

    system["system"]["solution"][
        "extrinsic_constraints"
    ]["density"]["c"] = float(rho_c)

    # --------------------------------------------------------
    # Evaluate thermodynamic state
    # --------------------------------------------------------
    
    
    tmp = evaluate_canonical_state(
        ctx=ctx,
        config_dict=system,
        fe_res=free_energy,
        supplied_data=None,
        export=False,
        verbose=False,
    )
    
    
    #print ("here is the value !!!!!!!!!", rho_a, rho_b, rho_c, tmp["free_energy_density"], "\n\n\n\n")
    
    

    # --------------------------------------------------------
    # Restore original densities
    # --------------------------------------------------------
    # Return free-energy density
    # --------------------------------------------------------

    return tmp["free_energy_density"]
# ============================================================
# XY PLANE
# ============================================================

export_data_XY = []

x_sum = p * rho_liquid[0] + (1-p) * rho_gas[0]
y_sum = p * rho_liquid[1] + (1-p) * rho_gas[1]

limx = max(rho_liquid[0], rho_gas[0], 2.005*x_sum)
limy = max(rho_liquid[1], rho_gas[1], 2.005*y_sum)

num_points = 100

x1_vals = np.linspace(1e-4, limx, num_points)
y1_vals = np.linspace(1e-4, limy, num_points)

X1, Y1 = np.meshgrid(x1_vals, y1_vals)

x2_vals = (x_sum - x1_vals*p)/(1.0-p)
y2_vals = (y_sum - y1_vals*p)/(1.0-p)

X2, Y2 = np.meshgrid(x2_vals, y2_vals)

mask1 = (X2 >= 0) & (Y2 >= 0)

Total_Energy_XY = np.full_like(
    X1,
    np.nan,
    dtype=np.float64
)

# ============================================================
# Compute XY free-energy surface
# ============================================================

valid_indices = np.argwhere(mask1)

with tqdm(
    total=len(valid_indices),
    desc="Computing XY surface"
) as pbar:

    for idx in valid_indices:

        i, j = idx

        

        omega1 = energy_free(
            X1[i,j],
            Y1[i,j],
            z_c_1
        )

        omega2 = energy_free(
            X2[i,j],
            Y2[i,j],
            z_c_2
        )

        total_energy = omega1*p + omega2*(1-p)

        # ----------------------------------------
        # IMPORTANT: clip unphysical energies
        # ----------------------------------------

        if total_energy < 2.5:

            Total_Energy_XY[i,j] = total_energy

            export_data_XY.append([
                X1[i,j],
                Y1[i,j],
                X2[i,j],
                Y2[i,j],
                total_energy
            ])

        
        pbar.update(1)

# ============================================================
# Save XY data
# ============================================================

np.savetxt(
    "output_XY_data.txt",
    export_data_XY,
    fmt="%.8f",
    header="X1 Y1 X2 Y2 TotalEnergy"
)

# ============================================================
# YZ PLANE
# ============================================================

export_data_YZ = []

y_sum_2 = p*rho_liquid[1] + (1-p)*rho_gas[1]
z_sum_2 = p*rho_liquid[2] + (1-p)*rho_gas[2]

limy2 = max(rho_liquid[1], rho_gas[1], 2.005*y_sum_2)
limz2 = max(rho_liquid[2], rho_gas[2], 2.005*z_sum_2)

y1_vals_2 = np.linspace(1e-4, limy2, num_points)
z1_vals_2 = np.linspace(1e-4, limz2, num_points)

Y1_2, Z1_2 = np.meshgrid(
    y1_vals_2,
    z1_vals_2
)

mask = Y1_2 < y_sum_2

Y2_2 = np.zeros_like(Y1_2)
Z2_2 = np.zeros_like(Z1_2)

Y2_2[mask] = (
    y_sum_2 - Y1_2[mask]*p
)/(1.0-p)

Z2_2[mask] = (
    z_sum_2 - Z1_2[mask]*p
)/(1.0-p)

Y2_2[~mask] = (
    y_sum_2 - Y1_2[~mask]*(1-p)
)/p

Z2_2[~mask] = (
    z_sum_2 - Z1_2[~mask]*(1-p)
)/p

mask2 = (Y2_2 >= 0) & (Z2_2 >= 0)

Total_Energy_YZ = np.full_like(
    Y1_2,
    np.nan,
    dtype=np.float64
)

# ============================================================
# Compute YZ free-energy surface
# ============================================================

valid_indices_2 = np.argwhere(mask2)

with tqdm(
    total=len(valid_indices_2),
    desc="Computing YZ surface"
) as pbar:

    for idx in valid_indices_2:

        i, j = idx

        try:

            if Y1_2[i,j] < y_sum_2:

                omega1 = energy_free(
                    x_c_1,
                    Y1_2[i,j],
                    Z1_2[i,j]
                )

                omega2 = energy_free(
                    x_c_2,
                    Y2_2[i,j],
                    Z2_2[i,j]
                )

                total_energy = omega1*p + omega2*(1-p)

            else:

                omega1 = energy_free(
                    x_c_2,
                    Y1_2[i,j],
                    Z1_2[i,j]
                )

                omega2 = energy_free(
                    x_c_1,
                    Y2_2[i,j],
                    Z2_2[i,j]
                )

                total_energy = omega1*(1-p) + omega2*p

            # ----------------------------------------
            # IMPORTANT: clip huge energies
            # ----------------------------------------

            if total_energy < 4.5:

                Total_Energy_YZ[i,j] = total_energy

                export_data_YZ.append([
                    Y1_2[i,j],
                    Z1_2[i,j],
                    Y2_2[i,j],
                    Z2_2[i,j],
                    total_energy
                ])

        except Exception as e:

            print(f"YZ failure at ({i},{j})")
            print(e)

        pbar.update(1)

# ============================================================
# Save YZ data
# ============================================================

np.savetxt(
    "output_YZ_data.txt",
    export_data_YZ,
    fmt="%.8f",
    header="Y1 Z1 Y2 Z2 TotalEnergy"
)

# ============================================================
# Remove garbage values
# ============================================================

Total_Energy_XY[
    Total_Energy_XY > 2.5
] = np.nan

Total_Energy_YZ[
    Total_Energy_YZ > 4.5
] = np.nan

# ============================================================
# PLOTTING
# ============================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble":
        r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
})

fig = plt.figure(figsize=(9,4.8))

outer = gridspec.GridSpec(
    1,
    2,
    width_ratios=[1,1],
    wspace=0.22
)

ax1 = fig.add_subplot(outer[0])

inner = gridspec.GridSpecFromSubplotSpec(
    1,
    2,
    subplot_spec=outer[1],
    width_ratios=[1,0.07],
    wspace=0.05
)

ax2 = fig.add_subplot(inner[0])
cax = fig.add_subplot(inner[1])

axs = [ax1, ax2]

# ============================================================
# Colormap normalization
# ============================================================

cmap = plt.cm.RdBu

vmin = np.nanmin(Total_Energy_XY)

# IMPORTANT FIX
vmax = vmin + 0.15

norm = Normalize(vmin=vmin, vmax=vmax)

levels = np.arange(vmin, vmax, 0.01)

# ============================================================
# XY plot
# ============================================================

cf1 = axs[0].contourf(
    X1,
    Y1,
    Total_Energy_XY,
    levels=levels,
    cmap=cmap,
    norm=norm,
    alpha=0.85
)

cl1 = axs[0].contour(
    X1,
    Y1,
    Total_Energy_XY,
    levels=levels,
    colors='black',
    linewidths=0.4
)

axs[0].scatter(
    rho_liquid[0],
    rho_liquid[1],
    facecolors='none',
    edgecolors='blue',
    marker='^',
    s=120,
    linewidths=2.0
)

axs[0].scatter(
    rho_gas[0],
    rho_gas[1],
    facecolors='none',
    edgecolors='green',
    marker='v',
    s=120,
    linewidths=2.0
)

axs[0].set_xlabel(r'$\rho_{\rm a}$', fontsize=22)
axs[0].set_ylabel(r'$\rho_{\rm b}$', fontsize=22)

axs[0].set_aspect('equal')

axs[0].set_title(r'(a)', fontsize=22)

# ============================================================
# YZ plot
# ============================================================

cf2 = axs[1].contourf(
    Y1_2,
    Z1_2,
    Total_Energy_YZ,
    levels=levels,
    cmap=cmap,
    norm=norm,
    alpha=0.85
)

cl2 = axs[1].contour(
    Y1_2,
    Z1_2,
    Total_Energy_YZ,
    levels=levels,
    colors='black',
    linewidths=0.4
)

axs[1].scatter(
    rho_liquid[1],
    rho_liquid[2],
    facecolors='none',
    edgecolors='blue',
    marker='^',
    s=120,
    linewidths=2.0
)

axs[1].scatter(
    rho_gas[1],
    rho_gas[2],
    facecolors='none',
    edgecolors='green',
    marker='v',
    s=120,
    linewidths=2.0
)

axs[1].set_xlabel(r'$\rho_{\rm b}$', fontsize=22)
axs[1].set_ylabel(r'$\rho_{\rm c}$', fontsize=22)

axs[1].set_aspect('equal')

axs[1].set_title(r'(b)', fontsize=22)

# ============================================================
# Colorbar
# ============================================================

cbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=cax
)

cbar.set_label(
    r'$\beta \mathcal{F}$',
    fontsize=18
)

# ============================================================
# Save figure
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.92,
    bottom=0.14,
    top=0.92
)

plt.savefig(
    "phase_diagram_combined_new.png",
    dpi=600,
    bbox_inches='tight'
)

plt.show()

print("\nPlots exported.\n")
