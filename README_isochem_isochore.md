# 🧮 README — Input File Guide for `isochem` / `isochore` Coexistence Solver

This document explains the structure and meaning of each section in the input file used for canonical (`isochore`) and grand canonical (`isochem`) coexistence density calculations in the CDFT-based solver.

---

## ⚙️ 1. Preliminaries

These parameters define the **ensemble type**, **mean-field method**, and **simulation task**.

| Keyword | Description | Allowed Values | Example |
|----------|--------------|----------------|----------|
| `ensemble` | Specifies the statistical ensemble type. Determines whether the solver operates in canonical (fixed density) or grand canonical (fixed chemical potential) mode. | `isochore`, `isochem` | `ensemble = isochem` |
| `method` | Chooses the theoretical treatment of the mean-field functional. | `emf`, `smf`, `joe` | `method = emf` |
| `task` | Defines the type of calculation to perform. | `inhomogeneous`, `bulk` | `task = inhomogeneous` |
| `space_dimension` | Dimension of simulation space. | `1`, `2`, `3` | `space_dimension = 1` |
| `space_confinement` | Type of confining geometry used in the simulation. | `abox`, `pcylinder`, `acylinder`, `asphere` | `space_confinement = abox` |

---

## 🧩 2. Particle and Interaction Setup

These parameters describe **species** and their **interactions**.

| Keyword | Description | Example |
|----------|--------------|----------|
| `particle_types` | Defines how particles are represented. | `particle_types = "cgrained"` |
| `species` | List of species names separated by commas. | `species = a, b, c` |

### Interaction Definitions

Each interaction line follows this pattern:

```
interaction: <pair> = <potential_type>, sigma = <σ>, cutoff = <r_cut>, epsilon = <ε>, [optional parameters...]
```

Example:

```
interaction: ab = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
interaction: ac = ma, sigma = 1.0, cutoff = 3.5, epsilon = 0.1787, m = 12, n = 6, lambda = 0.477246
```

Supported potential types include:
- `gs`: Gaussian soft-core  
- `ghc`: Gaussian hard-core  
- `hc`: Hard-core  
- `ma`: Mie-analytic form  
- `custom_3`: Custom user-defined form  

---

## 🧱 3. Confinement and Boundary Conditions

Defines the simulation box and wall parameters.

| Keyword | Description | Example |
|----------|--------------|----------|
| `box_extension` | Physical size of the simulation domain along each axis (Å). | `box_extension = 60, 25, 60` |
| `box_points` | Number of grid points in each direction. | `box_points = 200, 100, 100` |
| `aperiodicity_blocker` | Keyword for wall boundary condition. | `aperiodicity_blocker: wall` |
| `wall` | Defines particle type and wall positions. | `wall: particles = d`<br>`wall: position = (0,0,0), (60,0,0)` |
| `wall_interaction` | Interaction of wall with each species. | `wall_interaction: ad = custom_3, sigma = 1.0, cutoff = 3.2, epsilon = 0.1` |

---

## ⚗️ 4. Thermodynamic Properties

This section specifies the **intrinsic constraints** that define the system's state.  
It differs depending on the **ensemble**:

### (a) For `ensemble = isochem` (Grand Canonical)
Define chemical potentials and/or pressure:

```bash
chemical_potential: a = 16.0
chemical_potential: c = -1.2
pressure = 6.0
```

Stored as:

```json
"intrinsic_constraints": {
    "chemical_potential": {
        "a": 16.0,
        "c": -1.2
    },
    "pressure": 6.0
}
```

### (b) For `ensemble = isochore` (Canonical)
Define **average species fractions**:

```bash
species_fraction: a = 0.27
species_fraction: b = 0.27
species_fraction: c = 0.27
```

Stored as:

```json
"intrinsic_constraints": {
    "species_fraction": {
        "a": 0.27,
        "b": 0.27,
        "c": 0.27
    }
}
```

---

## 🧠 5. Solution Parameters

| Keyword | Description | Example |
|----------|--------------|----------|
| `number_of_phases` | Number of coexisting phases. | `number_of_phases = 2` |
| `heterogeneous_pair` | Pair of species that should *not* dominate the same phase. | `heterogeneous_pair = ab` |
| `total_density_bound` | Maximum allowed density (for stability). | `total_density_bound = 2.0` |

> **Note:** The number of heterogeneous pairs must be ≤ `number_of_phases - 1`.  
> Exceeding this limit may cause numerical instability or unsolvable systems.

---

## 🔬 6. Units and Physical Scales

| Keyword | Description | Example |
|----------|--------------|----------|
| `temperature` | Temperature (K). | `temperature = 300.0` |
| `length` | Characteristic system length (Å). | `length = 200.0` |

---

## 🧮 7. Solver Control Parameters

| Keyword | Description | Example |
|----------|--------------|----------|
| `alpha` | Mixing parameter for Picard iteration. | `alpha = 0.001` |
| `iteration_max` | Maximum iteration count. | `iteration_max = 20000` |
| `log_period` | Logging frequency (iterations). | `log_period = 20` |

---

## 🧾 8. Notes and Recommendations

- For `number_of_phases = n`, define up to `n–1` heterogeneous pairs.
- Avoid overconstraining: intrinsic constraints > `N_species - 1` → solver will exit with an error.
- For `isochore` runs, total of all `species_fraction` values should ≤ 1.0.
- Keep `total_density_bound` modest (2–5) to avoid divergence.

---

## 📁 9. Example Workflow

1. Prepare your input file as shown.
2. Generate the JSON file:

   ```bash
   python -m cdft_package.isochem.data_generators.solution_initiator your_input.inp
   ```

3. Check generated file:

   ```bash
   scratch/input_data_solution_initiator.json
   ```

4. Run the coexistence solver.

---
