EPSILON = 1e-6

def wall_potential_visualization(ctx):
    """
    Compute and visualize wall-particle V(r) and V(k) and per-species external potentials.

    Expects:
      - ctx.scratch_dir  (Path-like)
      - ctx.plots_dir    (Path-like)
      - scratch must contain 'input_data_space_confinement_parameters.json'
      - scratch must contain 'input_data_particles_interactions_parameters.json'
      - optional: scratch/supplied_data_r_space.txt (global isotropic r-grid)
      - optional: positions file (supplied_data_positions.xyz or supplied_data_r_space.txt used as fallback)
    """
    import numpy as np
    from scipy.integrate import simpson
    from scipy.special import spherical_jn
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path

    # centralized isotropic potential factory (returns vectorized V(r) callable)
    from cdft_solver.generators.potential.generator_pair_potential_isotropic import pair_potential_isotropic as ppi

    scratch = Path(ctx.scratch_dir)
    plots = Path(ctx.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_float_if_possible(x):
        try:
            return float(x)
        except Exception:
            return x

        
        
    def bessel_fourier_transform(r, V_r, k_max=10.0):
        k_space = np.linspace(0, 10, len(r))
        V_k = np.zeros_like(k_space)
        base = V_r * r**2
        for i, k in enumerate(k_space):
            integrand = 4 * np.pi * base * spherical_jn(0, k * r)
            V_k[i] = simpson(y=integrand, x=r)
        return V_k, k_space

    def read_positions_file(filename):
        """Read 3-column positions file (x,y,z). Returns Nx3 numpy array or None."""
        try:
            arr = np.loadtxt(filename)
            if arr.ndim == 1:
                # If it's a 1D array of positions (single point)
                if arr.size == 3:
                    arr = arr.reshape(1, 3)
                else:
                    raise ValueError("Position file must contain 3 columns (x,y,z).")
            if arr.shape[1] != 3:
                raise ValueError("Position file must contain 3 columns (x,y,z).")
            return np.array(arr, dtype=float)
        except Exception as e:
            print(f"[WARN] Could not read positions from '{filename}': {e}")
            return None

    def read_r_space_file(filename):
        """Read a 1D r-space vector from a file. Returns 1D numpy array or None."""
        try:
            r = np.loadtxt(filename)
            if r.ndim > 1:
                r = r.flatten()
            r = np.array(r, dtype=float)
            if r.size < 3:
                raise ValueError("r-space must contain at least 3 points.")
            # ensure non-decreasing
            if not np.all(np.diff(r) >= 0):
                r = np.sort(r)
            return r
        except Exception as e:
            print(f"[INFO] no usable r-space file '{filename}': {e}")
            return None

    def read_json(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read JSON '{filename}': {e}")
            return None

    def normalize_walls_properties(conf):
        """
        Accepts conf = parsed json dict of 'space_confinement_parameters'
        Supports either:
           - old style keys: walls_particles_type, walls_position, walls_normals, walls_interactions
           - new style keys: particles, positions, normals, interactions
        Returns normalized tuple:
           (walls_particles, walls_positions, walls_normals, walls_interactions)
        where walls_particles is a list of strings,
              walls_positions is list of [x,y,z],
              walls_normals is list of [nx,ny,nz],
              walls_interactions is dict with keys primary/secondary/tertiary
        """
        wp = conf.get("walls_properties", {})
        # particles
        walls_particles = wp.get("walls_particles_type") or wp.get("particles") or []
        # positions / normals
        walls_positions = wp.get("walls_position") or wp.get("positions") or []
        walls_normals = wp.get("walls_normals") or wp.get("normals") or []
        # interactions
        walls_interactions = wp.get("walls_interactions") or wp.get("interactions") or {"primary": {}, "secondary": {}, "tertiary": {}}

        # ensure lists and shapes
        if isinstance(walls_particles, str):
            walls_particles = [walls_particles]
        walls_positions = list(walls_positions)
        walls_normals = list(walls_normals)

        # ensure interaction levels exist
        for level in ("primary", "secondary", "tertiary"):
            walls_interactions.setdefault(level, {})

        return walls_particles, walls_positions, walls_normals, walls_interactions

    def build_Vr_from_entry(entry):
        """
        entry: dict containing at least type,sigma,epsilon,cutoff (cutoff optional)
        Returns (V_func, params_dict)
        """
        p_dict = entry.copy() if isinstance(entry, dict) else {}
        p_dict.setdefault("cutoff", 5.0)
        # allow numeric fields to be floats
        for k in ("sigma", "epsilon", "cutoff"):
            if k in p_dict:
                p_dict[k] = _to_float_if_possible(p_dict[k])
        V_func = ppi(p_dict)
        return V_func, p_dict

    # -----------------------------
    # Core: compute per-level potentials V(r) and V(k)
    # -----------------------------
    def calculate_interactions_visual(json_conf_file, global_r_space=None):
        data = read_json(json_conf_file)
        if data is None:
            raise FileNotFoundError(f"Could not load confinement JSON '{json_conf_file}'")
        sc = data.get("space_confinement_parameters", {})
        walls_particles, walls_positions, walls_normals, walls_interactions = normalize_walls_properties(sc)

        levels = ["primary", "secondary", "tertiary"]
        all_r_space = {L: {} for L in levels}
        all_V_r = {L: {} for L in levels}
        all_k_space = {L: {} for L in levels}
        all_V_k = {L: {} for L in levels}

        # iterate levels and pairs
        for level in levels:
            level_dict = walls_interactions.get(level, {}) or {}
            for pair_name, entry in level_dict.items():
                V_func, p_pars = build_Vr_from_entry(entry)

                itype = str(p_pars.get("type", "")).lower()
                sigma = float(p_pars.get("sigma", 0.0))
                cutoff = float(p_pars.get("cutoff", 5.0))

                # choose r_space
                if global_r_space is not None:
                    mask = (global_r_space >= 0.0) & (global_r_space <= cutoff)
                    if np.any(mask):
                        r_space = global_r_space[mask]
                    else:
                        r_space = np.linspace(0.0, cutoff, 1000)
                else:
                    # r_start logic similar to previous
                    if itype in ("gs", "wca", "ma") or "custom" in itype:
                        r_start = 0.0
                    else:
                        if itype == "lj":
                            r_start = sigma * 2.0 ** (1.0 / 6.0)
                        elif itype == "hc":
                            r_start = max(0.0, sigma - 0.5)
                        else:
                            r_start = sigma
                    r_space = np.linspace(r_start, cutoff, 1000)

                V_r = V_func(r_space)
                V_k, k_space = bessel_fourier_transform(r_space, V_r)

                all_r_space[level][pair_name] = r_space
                all_V_r[level][pair_name] = V_r
                all_k_space[level][pair_name] = k_space
                all_V_k[level][pair_name] = V_k

        # plotting overview V(r)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
        line_styles = ['-', '--', '-.', ':']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for level in levels:
            for i, pair in enumerate(all_r_space[level].keys()):
                ax.plot(all_r_space[level][pair], all_V_r[level][pair],
                        label=f"{pair} ({level})",
                        linewidth=2.0,
                        linestyle=line_styles[i % len(line_styles)],
                        color=colors[i % len(colors)])
        ax.set_title("Wall-particle potential V(r) (all levels)")
        ax.set_xlabel("r")
        ax.set_ylabel("V(r)")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(-10, 10)
        plt.tight_layout()
        r_plot_file = plots / "vis_walls_particles_interaction_potential_r.png"
        plt.savefig(r_plot_file, dpi=300)
        plt.close(fig)

        # plotting overview V(k)
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
        for level in levels:
            for i, pair in enumerate(all_k_space[level].keys()):
                ax2.plot(all_k_space[level][pair], all_V_k[level][pair],
                         label=f"{pair} ({level})",
                         linewidth=1.5,
                         linestyle=line_styles[i % len(line_styles)])
        ax2.set_title("Wall-particle potential V(k) (Bessel transform)")
        ax2.set_xlabel("k")
        ax2.set_ylabel("V(k)")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        k_plot_file = plots / "vis_walls_particles_interaction_potential_k.png"
        plt.savefig(k_plot_file, dpi=300)
        plt.close(fig2)

        return all_r_space, all_V_r, all_k_space, all_V_k, (walls_particles, walls_positions, walls_normals)

    # -----------------------------
    # Run calculation, reading files
    # -----------------------------
    json_file_path = scratch / 'input_data_space_confinement_parameters.json'
    if not json_file_path.exists():
        print(f"[ERROR] Confinement JSON not found: {json_file_path}")
        return 1

    # load optional global r-space
    global_r_file = scratch / 'supplied_data_r_space.txt'
    global_r_space = read_r_space_file(global_r_file) if global_r_file.exists() else None

    try:
        all_r_space, all_V_r, all_k_space, all_V_k, walls_meta = calculate_interactions_visual(json_file_path, global_r_space=global_r_space)
    except Exception as e:
        print(f"[ERROR] while calculating interactions: {e}")
        return 1

    walls_particles, walls_positions, walls_normals = walls_meta

    # -----------------------------
    # positions for external potential computation
    # -----------------------------
    # prefer a proper positions file; many older pipelines used supplied_data_r_space.txt
    pos_file_candidates = [
        scratch / 'supplied_data_positions.xyz',
        scratch / 'supplied_data_r_space.txt',   # fallback
        scratch / 'supplied_data_positions.txt'
    ]
    positions = None
    for p in pos_file_candidates:
        if p.exists():
            positions = read_positions_file(p)
            if positions is not None:
                break

    if positions is None:
        print("[WARN] positions file missing or invalid; aborting profile plotting.")
        return 1

    # load species from particles interactions json
    particles_json = scratch / "input_data_particles_interactions_parameters.json"
    prop = read_json(particles_json)
    if prop is None:
        print("[WARN] particle interaction JSON missing; aborting profile plotting.")
        return 1
    species_data = prop.get("particles_interactions_parameters", {}).get("species", [])

    # reload confinement json to extract walls_properties in normalized naming
    conf_json = read_json(json_file_path)
    sc = conf_json.get("space_confinement_parameters", {})
    walls_particles_norm, walls_positions_norm, walls_normals_norm, walls_interactions_norm = normalize_walls_properties(sc)

    # normalize walls_particles for indexing per-wall: if single flat list, treat as same for all positions
    walls_particles_used = walls_particles_norm
    if isinstance(walls_particles_used, list) and len(walls_particles_used) > 0 and len(walls_particles_used) != len(walls_positions_norm):
        # interpret as single particle types list applying to all walls
        walls_particles_per_wall = [walls_particles_used for _ in walls_positions_norm]
    else:
        # interpret as a per-wall list-of-lists if necessary
        walls_particles_per_wall = []
        for entry in (walls_particles_used or []):
            if isinstance(entry, list):
                walls_particles_per_wall.append(entry)
            else:
                walls_particles_per_wall.append([entry])

    # accumulate external potential per species
    xs = positions[:, 0]    # projection along x for plotting
    v_Ext_total = np.zeros_like(xs)
    total_potential_list = []
    v_ext_species = {}

    for specimen in species_data:
        v_ext_cast = np.zeros_like(xs)
        # for each wall position
        for i, wpos in enumerate(walls_positions_norm):
            ref_point = np.array(wpos, dtype=float)
            r_space_local = np.linalg.norm(positions - ref_point, axis=1)

            wall_particles_here = walls_particles_per_wall[i] if i < len(walls_particles_per_wall) else []
            if isinstance(wall_particles_here, str):
                wall_particles_here = [wall_particles_here]

            for level in ("primary", "secondary", "tertiary"):
                level_dict = walls_interactions_norm.get(level, {}) if 'walls_interactions' in sc or 'interactions' in sc else walls_interactions_norm.get(level, {})
                # iterate particle types present at this wall
                for wp_particle in wall_particles_here:
                    name1 = (wp_particle + specimen).strip()
                    name2 = (specimen + wp_particle).strip()
                    entry = None
                    if name1 in level_dict:
                        entry = level_dict[name1]
                    elif name2 in level_dict:
                        entry = level_dict[name2]
                    if entry is None:
                        continue
                    # build potential and evaluate on geometric distances
                    V_func = ppi(entry)
                    V_r = V_func(r_space_local)
                    v_ext_cast += V_r
                    v_Ext_total += V_r
                    total_potential_list.append(V_r)

        v_ext_species[specimen] = np.array(v_ext_cast)

    total_potential_list.append(v_Ext_total)
    total_potential = np.array(total_potential_list)

    print("\n\n... external potential due to wall particles exported successfully ...\n\n")

    # -----------------------------
    # Plot per-species potential profiles vs position magnitude
    # -----------------------------
    positions_magnitude = np.linalg.norm(positions, axis=1)

    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(12, 8), dpi=300)
    for i, (specimen, profile) in enumerate(v_ext_species.items()):
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        plt.plot(positions_magnitude, profile, marker='o', linestyle=style, color=color, label=f'Profile {specimen} and wall')
    plt.xlabel('Position Magnitude')
    plt.ylabel('Potential (v_Ext)')
    plt.title('Position vs Potential for Multiple Profiles')
    plt.grid(True)
    plt.ylim(-6, 10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / 'vis_walls_position_vs_potential_individual.png', dpi=300)
    plt.close()

    # -----------------------------
    # 3D scatter of potential over positions
    # -----------------------------
    try:
        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=v_Ext_total, cmap='viridis')
        plt.colorbar(sc, label='Potential (v_Ext)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        plt.title('3D Scatter Plot of Potential vs Positions')
        plt.tight_layout()
        plt.savefig(plots / 'vis_walls_potential_3d.png', dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not create 3D scatter (maybe positions are 1D/2D): {e}")

    print("wall_potential_visualization completed successfully.")
    return 0

# end wall_potential_visualization

