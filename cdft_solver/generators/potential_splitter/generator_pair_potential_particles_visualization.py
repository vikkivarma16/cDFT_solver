# density_functional_minimizer/pair_potentials.py

def pair_potential_particles_visualization(ctx):
    """
    Calculate and visualize particle-particle interaction potentials in r- and k-space.
    """

    # print ("I am running")
    import numpy as np
    from scipy.integrate import simpson
    from scipy.special import spherical_jn
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path

    # Import your centralized pair potential function
    from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi

    scratch_dir = Path(ctx.scratch_dir)
    plots_dir = Path(ctx.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # print("I am also running")
    # -----------------------------
    # Bessel Fourier transform
    # -----------------------------
    def bessel_fourier_transform(r, V_r):
        k_space = np.linspace(0, 10, len(r))
        V_k = np.zeros_like(k_space)
        base = V_r * r**2
        for i, k in enumerate(k_space):
            integrand = 4 * np.pi * base * spherical_jn(0, k * r)
            V_k[i] = simpson(y=integrand, x=r)
        return V_k, k_space

    # -----------------------------
    # Read particle data from JSON
    # -----------------------------
    def read_particle_data(json_file, level):
        with open(json_file, "r") as f:
            data = json.load(f)
        level_data = data["particles_interactions_parameters"]["interactions"][level]
        return level_data  # full dict per pair

    # -----------------------------
    # Main calculation loop
    # -----------------------------
    def calculate_interactions(json_file):
        levels = ["primary", "secondary", "tertiary"]
        all_r_space, all_V_r, all_k_space, all_V_k = {}, {}, {}, {}

        for level in levels:
            all_r_space[level], all_V_r[level], all_k_space[level], all_V_k[level] = {}, {}, {}, {}
            level_data = read_particle_data(json_file, level)

            for pair, values in level_data.items():
                # Build dictionary for pair_potential_isotropic
                potential_dict = values.copy()  # assumes keys: type, sigma, epsilon, cutoff, m, n, lambda
                V_func = ppi(potential_dict)   # get vectorized potential function

                r_start = 0.0 if potential_dict["type"] in ["gs", "wca", "ma"] or "custom" in potential_dict["type"] else potential_dict["sigma"]
                r_space = np.linspace(r_start, potential_dict.get("cutoff", 5.0), 1000)

                V_r = V_func(r_space)
                V_k, k_space = bessel_fourier_transform(r_space, V_r)

                all_r_space[level][pair] = r_space
                all_V_r[level][pair] = V_r
                all_k_space[level][pair] = k_space
                all_V_k[level][pair] = V_k

        # -----------------------------
        # Plot potentials
        # -----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(19, 8), dpi=100)
        line_styles = ["-", "--", "-.", ":"]
        colors = ["b", "g", "r", "c", "m", "y", "k"]

        for i, level in enumerate(levels):
            for j, pair in enumerate(all_r_space[level]):
                axes[0].plot(
                    all_r_space[level][pair],
                    all_V_r[level][pair],
                    label=f"{pair} {level}",
                    linewidth=2.5,
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)],
                )
                axes[1].plot(
                    all_k_space[level][pair],
                    all_V_k[level][pair],
                    label=f"{pair} {level}",
                    linewidth=2.5,
                    linestyle=line_styles[j % len(line_styles)],
                    color=colors[j % len(colors)],
                )

        axes[0].set_title("Potential V(r) for all interaction pairs")
        axes[0].set_xlabel("r")
        axes[0].set_ylabel("V(r)")
        axes[0].set_ylim(-5,5)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title("Potential V(k) for all interaction pairs (Bessel Fourier Transform)")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("V(k)/V(0)")
        axes[1].legend()
        axes[1].set_ylim(-50,50)
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / "vis_interaction_potentials.png", dpi=300)

        print("\nAll particle potentials visualized successfully.\n")

        return 0

    # -----------------------------
    # Execute
    # -----------------------------
    json_file_path = scratch_dir / "input_data_particles_interactions_parameters.json"
    return calculate_interactions(json_file_path)

