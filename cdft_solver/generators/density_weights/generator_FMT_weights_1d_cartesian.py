import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
EPSILON = 0.0000001  # Small value to avoid division by zero
PI = np.pi

def read_particle_data_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    interaction_types = {}
    closest_distances = {}
    
    for pair_type, values in data["particles_interactions_parameters"]["interactions"]["primary"].items():
        if values["type"] == "hc" or  values["type"] == "ghc":
            interaction_types[pair_type] = values["type"]
            closest_distances[pair_type] = values["sigma"]
            
    for pair_type, values in data["particles_interactions_parameters"]["interactions"]["secondary"].items():
        if values["type"] == "hc" or  values["type"] == "ghc":
            interaction_types[pair_type] = values["type"]
            closest_distances[pair_type] = values["sigma"]
            
    for pair_type, values in data["particles_interactions_parameters"]["interactions"]["tertiary"].items():
        if values["type"] == "hc" or  values["type"] == "ghc":
            interaction_types[pair_type] = values["type"]
            closest_distances[pair_type] = values["sigma"]
        
    return interaction_types, closest_distances

def identify_particle_types(interaction_types):
    particle_types = []
    for pair_type in interaction_types.keys():
        if pair_type[0] not in particle_types:
            particle_types.append(pair_type[0])
        if pair_type[1] not in particle_types:
            particle_types.append(pair_type[1])
    return particle_types

def calculate_particle_sizes(closest_distances, particle_types):
    particle_sizes = {}
    for particle_type in particle_types:
        interaction_key = particle_type * 2
        for pair_type, sigma in closest_distances.items():
            if interaction_key == pair_type:
                particle_sizes[particle_type] = closest_distances[interaction_key]
    return particle_sizes

def calculate_weight_function_k_space(particle_sizes, k_space, dimension):
    weight_functions = {}
    if dimension == 1:
        for particle_type, size in particle_sizes.items():
            weight_function = []
            for kx, ky, kz in k_space:
                weight_vector = [kx, ky, kz]
                k_value = kx
                mod_k = np.sqrt(k_value*k_value)
                #print("size is given by : ",size)
                if mod_k < EPSILON:
                    weight_vector.extend([
                        1,                       # Weight function at k=0
                        size * 0.5,              # Additional weight terms
                        PI * size**2 ,
                        PI * size**3 / 6.0 ,
                        0,                  # n1_x
                        0                   # n2_x
                    ])
                else:
                    weight_vector.extend([
                    
                        np.sin(k_value * PI * size) / (k_value * size * PI),
                        np.sin(k_value * PI * size) / (2.0 * k_value * PI),
                        size * np.sin(k_value * PI * size) / k_value,
                        (np.sin(k_value * PI * size) / (2.0 * k_value**3 * PI**2) - size * np.cos(k_value * PI * size) / (2.0 * k_value**2 * PI)),
                        1j*(k_value * PI * size * np.cos(k_value * PI * size) - np.sin(k_value * PI * size)) / (2.0 * size * PI**2 * k_value**2),
                                              # n1_y, n1_z
                        1j*(k_value * PI * size * np.cos(k_value * PI * size) - np.sin(k_value * PI * size)) / (k_value**2 * PI) 
                        
                         #1j*(1/96.0) * (-8*size**3.0*np.cos(size*PI*k_value)* PI**3 * k_value**3.0 + 40.0 * np.sin(size*PI*k_value)*(size*PI*k_value)**2.0 - 144*np.cos(size*PI*k_value)*size*PI*k_value + 144 * np.sin(size*PI*k_value) )/(PI**3 * size**2.0 * k_value**4)
                    ])
                    
                weight_function.append(np.array(weight_vector))
            weight_functions[particle_type] = np.array(weight_function)
    return weight_functions

def export_weight_functions_to_files(weight_functions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for particle_type, weight_function in weight_functions.items():
        print("I am running overlal")
        file_name = output_dir / f"supplied_data_weight_FMT_k_space_{particle_type}.txt"
        np.savetxt(file_name, weight_function)

def fmt_weights_1d(ctx):
    # File paths using ctx
    json_file_path = Path(ctx.scratch_dir) / 'input_data_particles_interactions_parameters.json'
    k_space_file_path = Path(ctx.scratch_dir) / 'supplied_data_k_space.txt'
    plot_dir = Path(ctx.plots_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Read interaction data
    interaction_types, closest_distances = read_particle_data_from_json(json_file_path)
    particle_types = identify_particle_types(interaction_types)
    particle_sizes = calculate_particle_sizes(closest_distances, particle_types)
    
    # Load k-space
    k_space = np.loadtxt(k_space_file_path)
    k_space = np.array(k_space)
    dimension = 1
    
    # Calculate weight functions
    weight_functions = calculate_weight_function_k_space(particle_sizes, k_space, dimension)
    
    # Export weights
    export_weight_functions_to_files(weight_functions, Path(ctx.scratch_dir))
    
    # Plot |weight| vs k
    k_vals = k_space[:,0]
    for particle_type, weight_function in weight_functions.items():
        Vk_abs = np.abs(weight_function[:,3])  # example: pick 4th component for plotting
        plt.figure(figsize=(7,5))
        plt.plot(k_vals, Vk_abs, label=f"|V(k)| {particle_type}")
        plt.xlabel("k")
        plt.ylabel("|V(k)|")
        plt.title(f"FMT Weight Function for {particle_type}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_dir / f"FMT_weight_{particle_type}.png", dpi=300)
        plt.close()

    print("\n\n... k space FMT weights calculated, exported, and plots saved heheeheh ... \n\n")

