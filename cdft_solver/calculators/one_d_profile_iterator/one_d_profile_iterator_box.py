def one_d_profile_iterator_box(ctx, config, export_json= True, export_plots = True, filename  = "one_d_profiles"):   



    """
    Required solvers (install via pip if needed). 
    For restricted systems, activate the environment first.
    """
    import numpy as np
    import json
    import math
    import matplotlib.pyplot as plt
    from pynufft import NUFFT
    from scipy.fft import fft, ifft
    import pyfftw.interfaces as fftw
    import sympy as sp
    from scipy.integrate import simpson
    from pathlib import Path
    from sympy import log, diff, lambdify
    from scipy import integrate
    from scipy.special import j0
            
    
    
    from cdft_solver.calculators.total_free_energy.total_free_energy import total_free_energy
    from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isocore import coexistence_densities_isocore
    from cdft_solver.calculators.coexistence_densities.calculator_coexistence_densities_isochem import coexistence_densities_isochem

    
    
    from cdft_solver.generators.potential_splitter.hc import hard_core_potentials
    from cdft_solver.generators.grids_properties.external_potential_grid import external_potential_grid
    from cdft_solver.generators.grids_properties.k_and_r_space_box import r_k_space_box
    from cdft_solver.generators.grids_properties.k_and_r_space_cylindrical import r_k_space_cylindrical
    from cdft_solver.calculators.free_energy_hard_core.hard_core_planer import hard_core_planer
    from cdft_solver.calculators.free_energy_mean_field.mean_field_planer import mean_field_planer
    from cdft_solver.generators.grids_properties.bulk_rho_mue_planer import bulk_rho_mue_planer
    
    
    
    scratch = Path(ctx.scratch_dir)
    plots = Path(ctx.plots_dir)
    system  =  config
    
    
    
    def find_key_recursive(d, key):
        if not isinstance(d, dict):
            return None
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                found = find_key_recursive(v, key)
                if found is not None:
                    return found
        return None
    
    
    
    r_k_grid = r_k_space_box(ctx=ctx, data_dict=system, export_json=True, filename="supplied_data_r_k_space_box.json")
    v_ext = external_potential_grid( ctx=ctx, data_dict=system, grid_properties=r_k_grid, export_json=True, filename="supplied_data_external_potential.json", plot=True )
    
    
    
    rdf_planer  =  find_key_recursive(config, "planer_rdf")
    planer_grid_config = {}
    planer_grid_config ["space_confinement_parameters"] = rdf_planer
    
    r_k_grid_planer = r_k_space_cylindrical(ctx = ctx,  data_dict =  planer_grid_config, export_json = True, filename = "supplied_data_r_k_space_box_planer.json")
    
    
    
    
    
    ensemble  =  find_key_recursive(config, "ensemble")
    filenames = {}
    filenames["hard_core"] = "supplied_data_free_energy_hard_core.json"
    filenames["mean_field"] = "supplied_data_free_energy_mean_field.json"
    filenames["ideal"] =  "supplied_data_free_energy_ideal.json"
    filenames["hybrid"] =  "supplied_data_free_energy_hybrid.json"
    
    
    
    hc_data = hard_core_potentials( ctx=ctx, input_data=config, grid_points=5000, file_name_prefix="supplied_data_potential_hc.json", export_files=True)
    free_energy  = total_free_energy(ctx=ctx, hc_data=hc_data, system_config=system, export_json=True, filenames = filenames)
    
    

    

    
    hc_free_energy_planer = hard_core_planer( ctx=ctx, hc_data=hc_data, export_json=False, filename="Solution_hardcore_z.json" )
    mf_free_energy_planer =  mean_field_planer( ctx=ctx, hc_data=hc_data, system_config=system, export_json=False, filename=None,)
    
   
    
    import sympy as sp

    def build_symbolic_free_energy_payload(
        hc_result,
        mf_result,
        hc_data,
    ):
        """
        Merge hard-core and mean-field symbolic results into a single payload
        suitable for activate_free_energy_symbolics().
        """

        species = hc_data["species"]
        n_species = len(species)

        # --------------------------------------------------
        # Density symbols
        # --------------------------------------------------
        rho_z  = [f"rho_{s}_z"  for s in species]
        rho_zs = [f"rho_{s}_zs" for s in species]

        # --------------------------------------------------
        # Weighted densities (already symbolic)
        # --------------------------------------------------
        variables = [
            [str(v) for v in var_list]
            for var_list in hc_result["variables"]
        ]

        # --------------------------------------------------
        # Pair interactions
        # --------------------------------------------------
        vij = [
            [f"v_{species[i]}_{species[j]}" for j in range(n_species)]
            for i in range(n_species)
        ]

        # --------------------------------------------------
        # Free-energy expressions
        # --------------------------------------------------
        FE_hc = hc_result["expression"]
        FE_mf = mf_result["expression"]

        FE_excess = sp.simplify(FE_hc + FE_mf)

        # --------------------------------------------------
        # Final payload
        # --------------------------------------------------
        payload = {
            "species": species,
            "sigma_eff": hc_data.get("sigma"),
            "flag": hc_data.get("flag"),

            "densities_z": rho_z,
            "densities_zs": rho_zs,

            "variables": variables,
            "vij": vij,

            "free_energy_mf_symbolic": str(FE_mf),
            "free_energy_hc_symbolic": str(FE_hc),
            "free_energy_excess_symbolic": str(FE_excess),
        }

        return payload
        
    symbolic_payload = build_symbolic_free_energy_payload(hc_result=hc_free_energy_planer,mf_result=mf_free_energy_planer,hc_data=hc_data,)

    
    
    
    
    
    
    
    
    
    
    
    # =====================================================================
    #  SYMBOLIC ACTIVATION USING THE RESULT OF run_module
    # =====================================================================

    def activate_free_energy_symbolics(result):
        """
        Takes the result from run_module()—which contains the exact dictionary
        you printed—and activates fully symbolic expressions for all free energies.
        Also loads species, sigma_eff, and flag arrays.
        """

        # -------------------------------------------------------
        # 1) Create density symbols
        # -------------------------------------------------------
        rho  = [sp.symbols(str(s)) for s in result["densities_z"]]
        rhos = [sp.symbols(str(s)) for s in result["densities_zs"]]

        # -------------------------------------------------------
        # 2) Weighted-density symbols n_{α,i}
        # -------------------------------------------------------
        n = []
        for species_vars in result["variables"]:
            n.append([sp.symbols(v) for v in species_vars])

        # -------------------------------------------------------
        # 3) v_ij symbols
        # -------------------------------------------------------
        vij = []
        for row in result["vij"]:
            vij.append([sp.symbols(str(v)) for v in row])

        # -------------------------------------------------------
        # 4) Create master symbol table
        # -------------------------------------------------------
        symbol_table = {}

        # densities
        for s in rho + rhos:
            symbol_table[str(s)] = s

        # weighted densities
        for i in range(len(n)):
            for a in range(6):
                symbol_table[str(n[i][a])] = n[i][a]

        # v_ij
        for i in range(len(vij)):
            for j in range(len(vij[i])):
                symbol_table[str(vij[i][j])] = vij[i][j]

        # -------------------------------------------------------
        # 5) Volume factors
        # -------------------------------------------------------

        # -------------------------------------------------------
        # 6) Parse free energy expressions
        # -------------------------------------------------------
        FE_mf     = sp.sympify(result["free_energy_mf_symbolic"], locals=symbol_table)
        FE_hc     = sp.sympify(result["free_energy_hc_symbolic"], locals=symbol_table)
        FE_excess = sp.sympify(result["free_energy_excess_symbolic"], locals=symbol_table)

        # -------------------------------------------------------
        # 7) Build FE object
        # -------------------------------------------------------
        class FEObject:
            pass

        FE = FEObject()

        # Symbolic fields
        FE.rho = rho
        FE.rhos = rhos
        FE.n = n
        FE.v = vij
        FE.mf = FE_mf
        FE.hc = FE_hc
        FE.excess = FE_excess
        FE.symbol_table = symbol_table

        # -------------------------------------------------------
        # 8) New: Physical species metadata from JSON
        # -------------------------------------------------------
        FE.species = result.get("species", None)
        FE.sigma_eff = result.get("sigma_eff", None)
        FE.flag = result.get("flag", None)

        return FE

        
        
    FE = activate_free_energy_symbolics(symbolic_payload)
    
    
    
    def build_mf_c1(FE, i):
        """
        Extract contributions of species i to all pair interactions (j,k):

            - rho1_part[j][k], rho2_part[j][k]

        After removing v_ij terms and splitting into rho1/rho2 dependencies.
        """

        N = len(FE.rho)

        # -------------------------------------------------------------
        # 1. Derivatives
        # -------------------------------------------------------------
        dF_dz  = sp.diff(FE.mf,  FE.rho[i])
        dF_dzs = sp.diff(FE.mf, FE.rhos[i])

        # -------------------------------------------------------------
        # 2. Define rho1 and rho2 symbols
        # -------------------------------------------------------------
        rho1 = [sp.symbols(f"rho_{j}_1") for j in range(N)]
        rho2 = [sp.symbols(f"rho_{j}_2") for j in range(N)]

        # -------------------------------------------------------------
        # 3. Substitution maps
        # -------------------------------------------------------------
        sub_A = {FE.rho[j]: rho1[j] for j in range(N)}
        sub_A.update({FE.rhos[j]: rho2[j] for j in range(N)})

        sub_B = {FE.rhos[j]: rho1[j] for j in range(N)}
        sub_B.update({FE.rho[j]: rho2[j] for j in range(N)})

        termA = sp.expand(dF_dz.subs(sub_A))
        termB = sp.expand(dF_dzs.subs(sub_B))

        # -------------------------------------------------------------
        # 4. Prepare output containers (N x N arrays)
        # -------------------------------------------------------------
        rho1_part_A = [[0]*N for _ in range(N)]
        rho2_part_A = [[0]*N for _ in range(N)]
        rho1_part_B = [[0]*N for _ in range(N)]
        rho2_part_B = [[0]*N for _ in range(N)]

        # -------------------------------------------------------------
        # 5. Process any additive expression
        # -------------------------------------------------------------
        A_terms = termA.args if termA.is_Add else [termA]
        B_terms = termB.args if termB.is_Add else [termB]

        # -------------------------------------------------------------
        # 6. Helper: strip v_ij and split rho1/rho2
        # -------------------------------------------------------------
        def process_term(t, out_rho1, out_rho2):
            for j in range(N):
                for k in range(N):
                    vij = FE.v[j][k]

                    # Only process terms containing this v_ij
                    if vij not in t.free_symbols:
                        continue

                    # Remove v_ij
                    t_no_vij = sp.simplify(t / vij)

                    # Split into rho1-only and rho2-only
                    t1, rest = t_no_vij.as_independent(*rho2)
                    t2, _    = rest.as_independent(*rho1)

                    out_rho1[j][k] += sp.simplify(t1)
                    out_rho2[j][k] += sp.simplify(t2)

        # -------------------------------------------------------------
        # 7. Apply splitting to A and B terms
        # -------------------------------------------------------------
        for t in A_terms:
            process_term(t, rho1_part_A, rho2_part_A)

        for t in B_terms:
            process_term(t, rho1_part_B, rho2_part_B)

        # -------------------------------------------------------------
        # 8. Return clean separated pieces
        # -------------------------------------------------------------
        
        
        
        
        
        return rho1_part_A, rho2_part_A, rho1_part_B, rho2_part_B



    # ============================================================
    # Build lambdifier for any of the four MF parts (N x N arrays)
    # ============================================================
    def lambdify_mf_part(expr_array, FE):
        """
        Lambdify an MF part expression (one of the four pieces):
            termA_rho1, termA_rho2, termB_rho1, termB_rho2

        Each part is an N x N array depending on rho1 and rho2 variables.
        """

        N = len(FE.rho)
        rho1_syms = [sp.symbols(f"rho_{i}_1") for i in range(N)]
        rho2_syms = [sp.symbols(f"rho_{i}_2") for i in range(N)]
        args = rho1_syms + rho2_syms

        # Flatten 2D array for lambdify
        result = [ ([sp.lambdify(args, expr_array[k][j], "numpy")  for j in range(N)]) for k in range(N)]
        


        return result

    # ============================================================
    # Build full MF c(1) database for all species and all parts
    # ============================================================
    c1_mf_parts = []  # store for each species i: [A_rho1_fn, A_rho2_fn, B_rho1_fn, B_rho2_fn]

    for i in range(len(FE.rho)):

        # Get filtered MF contributions (each term is N x N)
        termA_rho1, termA_rho2, termB_rho1, termB_rho2 = build_mf_c1(FE, i)

        # Lambdify each of the 4 pieces
        A1_fn = lambdify_mf_part(termA_rho1, FE)
        A2_fn = lambdify_mf_part(termA_rho2, FE)
        B1_fn = lambdify_mf_part(termB_rho1, FE)
        B2_fn = lambdify_mf_part(termB_rho2, FE)


        # Store all four parts for this species
        c1_mf_parts.append([A1_fn, A2_fn, B1_fn, B2_fn])







    # ============================================================
    # Build total MF free energy and factorize per v_ij
    # ============================================================
    def build_mf_total_factorized(FE):
        """
        Factorize the total mean-field free energy F_mf(1,2) into NxN
        v_ij terms, and split each term into rho1 and rho2 dependent parts.

        Returns:
            Fmf_12 : symbolic total F_mf(1,2)
            A       : NxN array, rho1 part of each v_ij term
            B       : NxN array, rho2 part of each v_ij term
        """
        N = len(FE.rho)
        
        # 1. rho1 and rho2 symbols
        rho1 = [sp.symbols(f"rho_{i}_1") for i in range(N)]
        rho2 = [sp.symbols(f"rho_{i}_2") for i in range(N)]
        
        # 2. Substitute rho_1 and rho_2 in F_mf
        sub = {FE.rho[j]: rho1[j] for j in range(N)}
        sub.update({FE.rhos[j]: rho2[j] for j in range(N)})
        Fmf_12 = sp.expand(FE.mf.subs(sub))
        
        # 3. Prepare NxN arrays
        A = [[0 for _ in range(N)] for _ in range(N)]
        B = [[0 for _ in range(N)] for _ in range(N)]
        
        # 4. Split into additive terms
        terms = Fmf_12.args if isinstance(Fmf_12, sp.Add) else [Fmf_12]
        
        # 5. Helper: strip v_ij and split rho1/rho2
        def process_term(t, i, j):
            vij = FE.v[i][j]
            if vij not in t.free_symbols:
                return 0, 0  # skip terms not containing this v_ij
            t_no_v = sp.expand(t / vij)
            t1, rest = t_no_v.as_independent(*rho2)
            t2, _    = rest.as_independent(*rho1)
            return sp.expand(t1), sp.expand(t2)
        
        # 6. Loop over all i,j to fill A and B
        for i in range(N):
            for j in range(N):
                for t in terms:
                    t1, t2 = process_term(t, i, j)
                    A[i][j] += t1
                    B[i][j] += t2
    
        #print("here is the matrix printed for the argument", A, "\n\n\n\n")
        #print("here is the matrix printed for the argument", B, "\n\n\n\n")
        
        return Fmf_12, A, B, sub

    # ============================================================
    # Lambdifier for NxN arrays
    # ============================================================
    def lambdify_mf_matrix(A, B, FE):
        N = len(FE.rho)
        rho1_syms = [sp.symbols(f"rho_{i}_1") for i in range(N)]
        rho2_syms = [sp.symbols(f"rho_{i}_2") for i in range(N)]
        args = rho1_syms + rho2_syms
        
        A_fn = [[sp.lambdify(args, A[i][j], "numpy") for j in range(N)] for i in range(N)]
        B_fn = [[sp.lambdify(args, B[i][j], "numpy") for j in range(N)] for i in range(N)]
        
        return A_fn, B_fn

    # ============================================================
    # USAGE EXAMPLE
    # ============================================================
    # Build symbolic factorized MF energy
    Fmf_12, A, B, submap = build_mf_total_factorized(FE)

    # Lambdify for numeric evaluation
    A_fn, B_fn = lambdify_mf_matrix(A, B, FE)

    # Now A[i][j] and B[i][j] correspond to rho1/rho2 factors of v_ij term



    

    def build_free_energy_and_derivatives_n_only(FE):
        """
        Build derivatives of total_phi w.r.t weighted densities only (n_{a,i}),
        and lambdify the derivatives and free energy as functions of n only.
        """
        phiz = FE.hc  
        # Flatten weighted densities into a list
        n_vars = [FE.n[i][a] for i in range(len(FE.n)) for a in range(6)]
        # Compute derivatives
        functions = [[sp.diff(phiz, FE.n[i][a]) for a in range(6)] for i in range(len(FE.n))]
        # Lambdify derivatives
        functions_func = [[sp.lambdify(n_vars, functions[i][a], 'numpy') for a in range(6)] for i in range(len(FE.n))]

        # Lambdify total free energy
        free_energy_func = sp.lambdify(n_vars, phiz, 'numpy')

        return functions_func, free_energy_func



    c1_hc_fn, Fhc_fn = build_free_energy_and_derivatives_n_only(FE)
    
    sigma_eff = FE.sigma_eff
    flag = FE.flag
    species  =  FE.species

    

    # --- Default values (in case needed) ---
    profile_p = find_key_recursive(config, "profile")
    alpha = find_key_recursive(profile_p, "alpha_mixing_max")
    iteration_max = find_key_recursive(profile_p, "iteration_max")
    log_period = find_key_recursive(profile_p, "log_period")
    tol = find_key_recursive(profile_p, "tolerance")



    params = find_key_recursive(system, "space_confinement_parameters")
    if params is None:
        raise KeyError("Could not find 'space_confinement_parameters' in the dictionary.")

    box_length = params["box_properties"]["box_length"]
    box_points = [int(p) for p in params["box_properties"]["box_points"]]
    dimension = int(params["space_properties"]["dimension"])


    
    if (ensemble == "isocore"):
        value = coexistence_densities_isocore( ctx = ctx, config_dict = config, fe_res = free_energy, supplied_data = None, max_outer_iters = 10, tol_outer = 1e-3,tol_solver = 1e-8, verbose = True)
    elif (ensemble == "isochem"):
        value = coexistence_densities_isochem( ctx = ctx, config_dict = config, fe_res = free_energy, supplied_data = None, max_outer_iters = 10, tol_outer = 1e-3,tol_solver = 1e-8, verbose = True)

    
       
    

    # Call the new bulk assignment
    rho_mue = bulk_rho_mue_planer(
        ctx=ctx,
        thermodynamic_parameter=value,
        r_space_coordinates=r_k_grid,
        export_json=True,
        filename="supplied_data_bulk_mue_rho_r_space.json",
        plot=False,
    )

    # ---------------------------------
    # Extract data directly (NO FILE IO)
    # ---------------------------------
    r_space = np.asarray(rho_mue["r_space"])     # (N,3)
    bulk_rhos = np.asarray(rho_mue["bulk_rhos"]) # (N, nspecies)
    bulk_mues = np.asarray(rho_mue["bulk_mues"]) # (N, nspecies)

    x = r_space[:, 0]
    y = r_space[:, 1]
    z = r_space[:, 2]

    # ---------------------------------
    # Reformat to match old expectations
    # rho_r[pid, :] and mue_r[pid, :]
    # ---------------------------------
    # Old code expected shape: (nspecies, N)
    rho_r = bulk_rhos.T
    mue_r = bulk_mues.T



    exit(0)




   



    k_space_file_path = scratch/ 'supplied_data_k_space.txt'
    k_space = np.loadtxt(k_space_file_path)
    k_space = np.array(k_space)

    kx = k_space[:,0]
    ky = k_space[:,1]
    kz = k_space[:,2]





    v_ext={}
    for key in species:
        with open(scratch / f"supplied_data_walls_potential_{key}_r_space.txt", "r") as file:
            v_ind=[]
            for line in file:
                # Skip comment lines
                if line.startswith("#"):
                    continue
                # Split the line into columns and convert them to floats
                columns = line.strip().split()
                v_ind.append( float(columns[3]))
            v_ext[key] = np.array(v_ind)
            
    print ("\n\n... supplied data has been imported successfully ...\n\n\n")








    # this is the main regions for the calculation for the 1 d walls confinement DFT simulation ...

    i = 0
    j = 0

    iteration = 0
    rho_r_initial = np.array(rho_r)
    rho_r_current = np.array(rho_r)

    piee = np.pi




    json_file_particles_interactions = scratch / "input_data_particles_interactions_parameters.json"
    json_file_simulation_thermodynamics = scratch / "input_data_simulation_thermodynamic_parameters.json"
    r_space_file = scratch / "supplied_data_r_space.txt"
    
    kx, ky, kz = [], [], []  # Define kx, ky, kz before the loop
    fmt_weights = {}  # Define fmt_weights to hold all species weights

    for key in species:
        fmt_weights_ind = []  # Initialize a list for individual species weights
        
        # Open file for the current species
        with open(scratch/ f"supplied_data_weight_FMT_k_space_{key}.txt", "r") as file:
            for line in file:
                # Skip comment lines
                if line.startswith("#"):
                    continue

                # Split the line into columns and convert them to floats
                columns = line.strip().split()
               
                
                # Collect rho-related values for this species
                li = []

                li.append(complex(columns[3]))
                li.append(complex(columns[4]))
                li.append(complex(columns[5]))
                li.append(complex(columns[6]))
                li.append(complex(columns[7]))
                li.append(complex(columns[8]))
           
                fmt_weights_ind.append(li)
                
        
        fmt_weights_ind = np.array (fmt_weights_ind)
        fmt_weights[key] = fmt_weights_ind # Append the individual weights list to fmt_weights


    threshold = 0.001

    print("\n\n... Total number of iteration is given as:", iteration_max, "\n\n\n\n\n")


    mf_weight = []
    
        
        
    for key_1 in species:
         mf_weight_in = []
         for key_2 in species:
            mf_weight_in.append(np.zeros(nx, dtype = complex))
            
         mf_weight.append(mf_weight_in)
        
    i = 0 
    for key in species:
         # Initialize a list for individual species weights
        j = 0
       
        for key_in in species:
            if (j >= i):
                
                # Open file for the current species
                with open(scratch / f"supplied_data_weight_MF_k_space_{key}{key_in}.txt", "r") as file:
                    k =0
                    for line in file:
                        # Skip comment lines
                        if line.startswith("#"):
                            continue

                        # Split the line into columns and convert them to floats
                        columns = line.strip().split()
                       
                        
                        # Collect rho-related values for this species
                    
                   
                        mf_weight[i][j][k] = complex(columns[0])
                        mf_weight[j][i][k] = complex(columns[0])
                        k = k+1
                
                
            
            j = j + 1
        # Append the individual weights list to fmt_weights
        i = i + 1




    pressure_values = np.zeros(nx)

    surface_tension_values = np.zeros(nx)


    fu = open(scratch / "data_surface_tension_vs_t.txt", "w")
    
    log_fu = open(scratch / "data_log_output.txt", "w")
    log_fu.close()
    
    print ("Till here, everything is fine !!!!!!!!!!!!!!\n\n\n")
    

    grand_rosenfeld_flag = 0
    for i in range (len(flag)):
        if (flag[i] == 1):
            grand_rosenfeld_flag = 1
            
    grand_meanfield_flag = 1
                

    def make_ideal_gas_pressure(FE):
        """
        Returns a lambdified ideal gas pressure function:
            p = sum_i rho_i

        The function takes a single argument:
            rho_vec  (array of species densities)
        """

        # symbolic variables rho_0, rho_1, ..., rho_{N-1}
        rho_syms = [sp.symbols(f"rho_{i}") for i in range(len(FE.rho))]

        # symbolic ideal gas pressure
        P_expr = sum(rho_syms)

        # lambdify -> numerical function taking N scalars
        P_num = sp.lambdify(rho_syms, P_expr, "numpy")

        # user-level function
        def pressure_fn(rho_vec):
            return P_num(*list(rho_vec))

        return pressure_fn

    

    func_pressure = make_ideal_gas_pressure(FE)


    temperature = 1.0
   

    

    while (iteration < iteration_max):
        
        rho_r_initial = rho_r_current 
        
        landau = np.zeros(nx) 
        # energy filtration for the hard core fmt terms ...
        rho_alpha_r={}
        fmt_flag = 0
        pid = 0
        
        if (grand_rosenfeld_flag == 1):
            pid = 0
            for particle in species:
                rho_k_ind = fft(rho_r_current[pid])
                omega_rho_k = np.zeros((6, nx), dtype=complex)
                li=[]
                for i in range(6):
                    omega_rho_k[i,:] = fmt_weights[particle][:, i] * rho_k_ind 
                    
                        #print(fmt_weights[particle1][:, i])
                   
                rho_alpha_r_ind= np.zeros((6,nx))
                for i in range(6):
                    rho_alpha_r_ind[i,:]= ifft(omega_rho_k[i, :]).real
        
                    
                rho_alpha_r[particle] = np.array(rho_alpha_r_ind)
                pid = pid + 1
            
            tdphi = []
            dphi=np.zeros((6, nx))
            
            pid = 0
            for particle in species:
                dphi = np.zeros((6, nx))
                tdphi.append(dphi)
        
            for k in range (nx):
                pid = 0
                for particle in species:
                    if ( (1.0 - rho_alpha_r[particle][3, k]) > 0.00000001 and (rho_alpha_r[particle][3, k]) > 0.00000001):
                        variable  = []
                        for particle_in in species:
                            variable.append(rho_alpha_r[particle_in][0, k])
                            variable.append(rho_alpha_r[particle_in][1, k])
                            variable.append(rho_alpha_r[particle_in][2, k])
                            variable.append(rho_alpha_r[particle_in][3, k])
                            variable.append(rho_alpha_r[particle_in][4, k])
                            variable.append(rho_alpha_r[particle_in][5, k])
                        
                        
                        tdphi[pid][0, k] =  c1_hc_fn[pid][0](*variable)
                        tdphi[pid][1, k] =  c1_hc_fn[pid][1](*variable)
                        tdphi[pid][2, k] =  c1_hc_fn[pid][2](*variable)
                        tdphi[pid][3, k] =  c1_hc_fn[pid][3](*variable)
                        tdphi[pid][4, k] =  c1_hc_fn[pid][4](*variable)
                        tdphi[pid][5, k] =  c1_hc_fn[pid][5](*variable)
                    pid = pid + 1
                landau[k] = Fhc_fn(*variable)
            
            pid = 0
            dphi_k_new = [] 
            for particle in species:
                dphi_k = []
                for i in range(6):
                    dphi_k_alpha = fft(tdphi[pid][i])
                    dphi_k.append(dphi_k_alpha)
                dphi_k = np.array(dphi_k)
                dphi_k_new.append(dphi_k)
                pid = pid + 1
                    

        
            pid = 0
            
            f_ext_frag = []
            for particle in species:
                dphi_k_ind = dphi_k_new[pid]
                omega_dphi_k = np.zeros((6, nx), dtype=complex)
                li=[]
                for i in range(6):
                    omega_dphi_k[i,:] = fmt_weights[particle][:, i] * dphi_k_ind[i] 
                    
                        #print(fmt_weights[particle1][:, i])
                   
                dphi_rho_alpha_ind= np.zeros((6,nx))
                for i in range(6):
                    dphi_rho_alpha_ind[i,:]= ifft(omega_dphi_k[i, :]).real
        
                    
                frag_energy = np.array(dphi_rho_alpha_ind)
                f_ext_frag.append(frag_energy)
                pid = pid + 1
      
            
            pid = 0
            
            total_df_ext = []
            for particle in species:
                
                df_ext_ind = np.zeros(nx)
                
                for i in range(nx):
                    df_ext_ind[i] = np.sum(f_ext_frag[pid][:, i])
                
                total_df_ext.append(df_ext_ind)
                pid = pid +1
            
    
        
        
        # print("\n\n\n\n!!!!!!!!!! ...I ran till here... !!!!!!!!!!\n\n\n\n", "!!!!!!!!!! ...landau potential is being printed now... !!!!!!!!!!", landau, "\n\n\n\n")
        
        
        if grand_meanfield_flag == 1:

            total_f_ext_mf = []

            N = len(species)  # number of species
            nx = len(rho_r_current[0])  # grid points

            # Pre-build lambdified density vectors per grid point
            for i in range(N):
                ind_mf_energy = np.zeros(nx)

                for k in range(N):
                    energy_r_ind = np.zeros(nx)

                    for j in range(N):

                        # Allocate arrays for FFT convolution
                        del_rhoA_1 = np.zeros(nx)
                        del_rhoA_2 = np.zeros(nx)
                        del_rhoB_1 = np.zeros(nx)
                        del_rhoB_2 = np.zeros(nx)

                        rho_1_factor = np.zeros(nx)
                        rho_2_factor = np.zeros(nx)

                        # Loop over spatial grid
                        for ldx in range(nx):
                            # Build densities vector for lambdified functions
                            #print (rho_r_current)
                            
                            
                            rho_point = rho_r_current[:, ldx]  # shape (N,)
                            densities = np.concatenate([rho_point, rho_point])
                            
                            
                            #print ("\n\n\n\n", rho_point, "this is the value which has been concatenated please rectify the same!!!!\n\n\n")
                            
                            #print (densities)
                            
                            #print ("\n\n\n\n")
                            
                            #exit(0)
                            
                        
                            # Evaluate N x N MF parts for this i,k,j
                            del_rhoA_1[ldx] = c1_mf_parts[i][0][k][j](*densities)
                            del_rhoA_2[ldx] = c1_mf_parts[i][1][k][j](*densities)
                            del_rhoB_1[ldx] = c1_mf_parts[i][2][k][j](*densities)
                            del_rhoB_2[ldx] = c1_mf_parts[i][3][k][j](*densities)

                            # Evaluate factorized A/B contributions
                            rho_1_factor[ldx] = A_fn[k][j](*densities)
                            rho_2_factor[ldx] = B_fn[k][j](*densities)
                            
                            
                        
                            
                        

                        # FFTs
                        fft_delA_2 = fft(del_rhoA_2)
                        fft_delB_2 = fft(del_rhoB_2)
                        fft_rho2 = fft(rho_2_factor)

                        # Multiply by MF weights
                        k_poden_A = fft_delA_2 * mf_weight[k][j]
                        k_poden_B = fft_delB_2 * mf_weight[k][j]

                        # Convolution for energy
                        energy_r_ind += del_rhoA_1 * ifft(k_poden_A).real + del_rhoB_1 * ifft(k_poden_B).real

                        # Optional Landau term (if i==0)
                        if i == 0:
                            k_poden = fft_rho2 * mf_weight[k][j]
                            landau += rho_1_factor * ifft(k_poden).real

                    ind_mf_energy += energy_r_ind

                total_f_ext_mf.append(ind_mf_energy)

    
        
        
        i = int(nx / 4)
        bulk_dens = []
        pid = 0
        for particle  in species:
            bulk_dens.append(rho_r_current[pid][i])        
            pid = pid +1
         
         
        j = 0
        

    
        
        for i in range(nx):
                
            pid = 0
            grand_landau = 0.0
            ind_density = []
            
            for particle in species:
                
                ind_density.append(rho_r_current[pid][i])
                free_energy = 0.0
                
                if (grand_rosenfeld_flag == 1):
                    free_energy = free_energy + total_df_ext[pid][i]
                if (grand_meanfield_flag == 1):
                    free_energy = free_energy + total_f_ext_mf[pid][i]
                  
                density = (np.exp( - v_ext[particle][i]/ temperature) * np.exp(mue_r[pid][i]) * np.exp( - free_energy) )
                grand_landau = grand_landau + (v_ext[particle][i]/ temperature - mue_r[pid][i]) * rho_r_current[pid][i] + rho_r_current[pid][i] *np.log(rho_r_current[pid][i]) -  rho_r_current[pid][i]
                rho_r_current[pid][i] = alpha * density + (1-alpha) * rho_r_initial[pid][i] 
                rho_r_initial[pid][i] = rho_r_current[pid][i]
                pid = pid + 1
                
                
            grand_landau += landau[i]
            surface_tension_values [i] =  func_pressure(bulk_dens) + grand_landau # - func_pressure(*ind_density) #
            pressure_values[i] = func_pressure(ind_density)
  
       
        i_start = int(2*nx / 6)
        i_end   = int(4*nx / 6)
        
        surface_tension_values = np.array(surface_tension_values)

        
        # Slice the relevant domain
        x_slice = x[i_start:i_end]
        f_slice = surface_tension_values[i_start:i_end]
        surface_tension_sum  = 2.0*np.sum(f_slice)/nx
        # Perform Simpson's rule integration
        total_surface_tension = simpson(f_slice, x_slice)
        fu.write(str (iteration))
        fu.write("  ")
        fu.write(str(total_surface_tension))
        fu.write("  ")
        fu.write(str(surface_tension_sum))
        fu.write("\n")
        
        if (iteration + 1) % log_period == 0:
            log_fu = open(scratch / "data_log_output.txt", "a")
            log_fu.write(f"--- Iteration {iteration + 1} ---\n")
            i=0
            for particle in species:
                free_energy = 0.0
                if grand_rosenfeld_flag == 1:
                    free_energy += total_df_ext[i][int(nx/2)]
                if grand_meanfield_flag == 1:
                    free_energy += total_f_ext_mf[i][int(nx/2)]

                line = f"inhomogeneous free energy per particle, for the species {particle}: {free_energy + np.log(rho_r_current[i][int(nx/2)]):.6e} " \
                       f"mue: {mue_r[i][int(nx/2)]:.6e} " \
                       f"current density at the center of the profile: {rho_r_current[i][int(nx/2)]:.6e}\n"

                print(line, end="")  # print to console
                log_fu.write(line)   # write to file
                i = i+1
            

            # Pressure and iteration info
            pressure_line = f"Pressure value is given as: {pressure_values[int(nx/2)]:.6e}\n\n"
            iteration_line = f"Number of iteration is given as: {iteration + 1}\n\n\n"
            print(pressure_line + iteration_line, end="")
            log_fu.write(pressure_line)
            log_fu.close()
            #log_fu.write(iteration_line)
        
        iteration =  iteration + 1 
       
    
    fu.close()



    file_name_pressure = scratch / f"data_pressure.txt"
    data = np.column_stack((x, pressure_values))
    np.savetxt(file_name_pressure, data)


    file_name_domega = scratch /  f"data_delta_omega.txt"
    data = np.column_stack((x, surface_tension_values))
    np.savetxt(file_name_domega, data)





    i=0
    for other_species in species:
        file_name = scratch / f"data_density_distribution_r_{other_species}.txt"
        
        data = np.column_stack((x, rho_r_current[i]))

        np.savetxt(file_name, data)
        i = i + 1

    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Plotting
    plt.figure(figsize=(12, 8), dpi=300)  # High-definition plot

    # Loop through each potential profile
    for i, key in enumerate(species):
        # Cycle through line styles and colors
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        plt.plot(x, rho_r_current[i], marker='o', linestyle=style, color=color, label=f'Species {key}')
        #plt.ylim(0, 0.004)

    # Plot customization
    plt.xlabel('Position Magnitude')
    plt.ylabel('Density distribution')
    plt.title('Density distribution for different species')
    plt.grid(True)
    plt.legend()

    # Save the plot in high resolution
    plt.savefig(plots / 'vis_rho_distribution.png')
