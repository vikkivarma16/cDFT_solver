````markdown
# cDFT_solver

A Python package for solving **classical Density Functional Theory (cDFT)** problems, with a strong focus on **pair-potential construction**, **interaction decomposition**, and **physics-consistent preprocessing pipelines**.

**Author:** Vikki Anand Varma  
**Email:** vikkivarma16@gmail.com  
**Background:** PhD in Physics (IIT Delhi). Specializes in computational physics, molecular simulations, and Python-based scientific tooling for interaction modeling, geometry, and visualization.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/vikkivarma16/cDFT_solver.git
````

**Requirements**

* Python ≥ 3.7
* NumPy, SciPy
* Standard scientific Python stack

---

## Overview

The package is organized around two core ideas:

1. **Flexible dictionary-based input parsing**
2. **Modular construction of pair interaction potentials**

All intermediate data structures are **JSON-serializable**, ensuring:

* reproducibility
* transparent preprocessing
* one-time conversion to NumPy arrays during runtime

---

# 1. Universal Dictionary Builder

## `super_dictionary_creator`

A **universal hierarchical dictionary generator** designed to convert structured text input files into fully nested Python dictionaries.

This module is the **entry point** for most workflows and feeds directly into the potential-generation pipeline.

---

### Import

```python
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
```

---

### Function Signature

```python
super_dictionary_creator(
    ctx=None,
    input_file=None,
    base_dict=None,
    export_json=False,
    filename="super_dictionary.json",
    super_key_name="system",
)
```

---

### Parameters

| Parameter        | Type       | Description                                                      |
| ---------------- | ---------- | ---------------------------------------------------------------- |
| `ctx`            | object     | Optional context providing `input_file` and `scratch_dir`        |
| `input_file`     | str / Path | Path to structured text input (required if `ctx` is None)        |
| `base_dict`      | dict       | Optional base dictionary whose attributes override parsed values |
| `export_json`    | bool       | Export resulting dictionary as JSON                              |
| `filename`       | str        | Output JSON filename                                             |
| `super_key_name` | str        | Top-level dictionary key                                         |

---

### Supported Input Format

Example structured input:

```
species = a, b, c
interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
profile iteration_max = 5000, tolerance = 1e-5, alpha = 0.1
```

**Features**

* Hierarchical parsing using `:` or whitespace
* Attribute assignment via `key = value`
* Multi-valued attributes supported
* Repeated keys promoted to lists
* Optional override via `base_dict`

---

### Output Example

```json
{
  "system": {
    "species": ["a", "b", "c"],
    "interaction": {
      "primary": {
        "aa": {"type": "gs", "sigma": 1.414, "cutoff": 3.5, "epsilon": 2.01},
        "ab": {"type": "gs", "sigma": 1.414, "cutoff": 3.5, "epsilon": 2.5}
      }
    },
    "profile": {
      "iteration_max": 5000,
      "tolerance": 1e-5,
      "alpha": 0.1
    }
  }
}
```

---

# 2. Pair Potential Generation API

This API implements a **complete isotropic pair-potential workflow**, designed for cDFT and particle-based theories.

All stages operate on **dictionaries**, not files, and optionally export JSON for caching or reuse.

---

## `pair_potential_isotropic(specific_pair_potential)`

### Description

Generates a **callable isotropic pair potential**
[
U(r)
]
from a single interaction dictionary.

This is the **core low-level generator** used internally by all higher-level modules.

---

### Parameters

| Name                      | Type | Description                                                         |
| ------------------------- | ---- | ------------------------------------------------------------------- |
| `specific_pair_potential` | dict | Interaction definition (`type`, `sigma`, `epsilon`, `cutoff`, etc.) |

---

### Returns

| Type     | Description                |
| -------- | -------------------------- |
| callable | Vectorized function `U(r)` |

---

### Notes

* Supports `hc`, `ghc`, `lj`, `mie`, `gs`, `ma`, etc.
* NumPy-compatible and vectorized
* No file I/O

---

## `hard_core_potentials(...)`

```python
def hard_core_potentials(
    ctx=None,
    data_dict=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_hc.json",
    export_files=True
)
```

### Description

Detects **hard-core interactions**, determines effective hard-core diameters, and constructs **purely repulsive hard-core potentials**.

This module:

* Identifies hard-core behavior pairwise
* Applies additive consistency rules
* Builds step-function HC potentials
* Optionally exports results as JSON

### Parameters

| Name               | Type             | Description                                |
| ------------------ | ---------------- | ------------------------------------------ |
| `ctx`              | object, optional | Provides `scratch_dir` for output          |
| `data_dict`        | dict             | Input interaction dictionary               |
| `grid_points`      | int              | Number of points in the radial grid        |
| `file_name_prefix` | str              | Output JSON filename                       |
| `export_files`     | bool             | If True, exports JSON to scratch directory |

### Returns

```python
{
  "species": [...],
  "sigma": [[...]],
  "flag": [[...]],
  "potentials": {
    "ij": {
      "r": [...],
      "U_hc": [...]
    }
  }
}
```

### Notes

* Hard-core potentials are **infinite below σ** and zero otherwise.
* Output is JSON-safe and NumPy-ready.
* No assumptions about dictionary nesting depth.

---

## `raw_potentials(...)`

```python
def raw_potentials(
    ctx=None,
    data_dict=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_raw.json",
    export_file=True
)
```

### Description

Computes **raw pair potentials** by summing **all interaction levels** (primary, secondary, tertiary) without any physical splitting.

[
U_{ij}^{raw}(r) = \sum_{\text{levels}} U_{ij}^{(level)}(r)
]

### Parameters

| Name               | Type             | Description                  |
| ------------------ | ---------------- | ---------------------------- |
| `ctx`              | object, optional | Used only if exporting       |
| `data_dict`        | dict             | Input interaction dictionary |
| `grid_points`      | int              | Radial grid resolution       |
| `file_name_prefix` | str              | Output JSON filename         |
| `export_file`      | bool             | Export JSON if True          |

### Returns

```python
{
  "potentials": {
    "ij": {
      "r": [...],
      "U_raw": [...]
    }
  }
}
```

### Notes

* No physical interpretation applied
* Useful for debugging and validation
* Output is JSON-serializable

---

## `meanfield_potentials(...)`

```python
def meanfield_potentials(
    ctx=None,
    data_dict=None,
    grid_points=5000,
    file_name_prefix="supplied_data_potential_mf.json",
    export_file=True
)
```

### Description

Constructs **mean-field interaction potentials** by applying conversion rules:

| Original Type | Mean-Field Replacement         |
| ------------- | ------------------------------ |
| `hc`, `ghc`   | zero potential                 |
| `lj`          | shifted-attractive LJ (`salj`) |
| `mie`         | attractive Mie (`ma`)          |

Hard-core contributions are removed by construction.

### Parameters

| Name               | Type             | Description                  |
| ------------------ | ---------------- | ---------------------------- |
| `ctx`              | object, optional | Used only if exporting       |
| `data_dict`        | dict             | Input interaction dictionary |
| `grid_points`      | int              | Radial grid resolution       |
| `file_name_prefix` | str              | Output JSON filename         |
| `export_file`      | bool             | Export JSON if True          |

### Returns

```python
{
  "species": [...],
  "mf_interactions": {...},
  "potentials": {
    "ij": {
      "r": [...],
      "U_mf": [...]
    }
  }
}
```

### Notes

* Mean-field potentials are **purely attractive**
* Designed for convolution-based DFT kernels
* Output remains JSON-safe

---

## `total_potentials(...)`

```python
def total_potentials(
    hc_source,
    mf_source,
    return_numpy=False
)
```

### Description

Assembles **total interaction potentials** by summing hard-core and mean-field contributions:

[
U_{ij}(r) = U_{ij}^{HC}(r) + U_{ij}^{MF}(r)
]

### Parameters

| Name           | Type         | Description                             |
| -------------- | ------------ | --------------------------------------- |
| `hc_source`    | dict or path | Hard-core potential dictionary or JSON  |
| `mf_source`    | dict or path | Mean-field potential dictionary or JSON |
| `return_numpy` | bool         | Return NumPy arrays if True             |

### Returns

```python
{
  "species": [...],
  "hc_potentials": {...},
  "mf_potentials": {...},
  "total_potentials": {
    "ij": {
      "r": [... or np.ndarray],
      "U_total": [... or np.ndarray]
    }
  }
}
```

### Notes

* Grid consistency is enforced
* Safe for direct numerical solvers
* Final assembly point before simulation

---

## 🔚 Summary

This API provides a **complete, modular, physics-consistent pipeline**:

1. Define pair interactions
2. Detect hard-core structure
3. Generate mean-field attractions
4. Assemble total potentials
5. Convert to NumPy only when needed

This design ensures:

* **Numerical stability**
* **Physical clarity**
* **Zero hidden coupling**
* **DFT-ready outputs**
