
````markdown
# cDFT_solver

A Python package to solver classical density functional theory based problems.

**Author:** Vikki Anand Varma  
**Email:** vikkivarma16@gmail.com  
**Bio:** PhD in Physics from IIT Delhi. Specializes in computational modeling, molecular simulations, and Python-based tools for 3D object manipulation, geometry operations, and visualization in scientific computing and materials modeling.

---

## Installation

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/vikkivarma16/cDFT_solver.git
````

Ensure that you have **Python 3.7+** and the necessary dependencies installed. Visualization requires **Paraview** or any **VTK-compatible viewer**.




Here’s a clear and concise README-style documentation for your `super_dictionary_creator` function:

---

# `super_dictionary_creator`

A **universal dictionary builder** from hierarchical input files for simulation or parameter configuration purposes.

It parses structured text files with support for:

* Hierarchy via colons (`:`) or whitespace
* Key-value attributes (`key = value`)
* Multi-value attributes (`sigma = 1.0, 1.1, 1.5`)
* Repeated keys (stored as lists)
* Promotion of attribute-only keys
* Optional base dictionary updates (overwrite attributes from `base_dict`)

---

## Installation / Import

```python
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator
```

---

## Function Signature

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

### Parameters

| Parameter        | Type       | Description                                                                                              |
| ---------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| `ctx`            | object     | Optional context with `ctx.input_file` and `ctx.scratch_dir`. Can be `None` if `input_file` is provided. |
| `input_file`     | str / Path | Path to the input text file to parse. Required if `ctx` is not given.                                    |
| `base_dict`      | dict       | Optional base dictionary whose keys/attributes will overwrite the parsed dictionary where applicable.    |
| `export_json`    | bool       | If `True`, the final dictionary is exported to JSON.                                                     |
| `filename`       | str        | Name of the exported JSON file (default: `"super_dictionary.json"`).                                     |
| `super_key_name` | str        | Top-level key for the dictionary (default: `"system"`).                                                  |

---

## Input File Format

Example of hierarchical input with key-value attributes:

```
species = a, b, c
interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
profile iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
```

* Hierarchies are split by `:` or spaces.
* Attribute names are always the last word to the left of `=`.
* Multi-value attributes are supported: `sigma = 1.0, 1.1, 1.5`.

---

## Output

Generates a nested dictionary preserving:

* Hierarchy
* Attribute-value mapping
* Repeated keys as lists
* Optional promotion of attribute-only keys

Example output:

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
    "profile": {"iteration_max": 5000, "tolerance": 0.00001, "alpha": 0.1}
  }
}
```

---

## Usage Example

```python
from types import SimpleNamespace
from pathlib import Path
from cdft_solver.generators.parameters.advance_dictionary import super_dictionary_creator

# Prepare input file
input_text = """
species = a, b, c
interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.01
interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
profile iteration_max = 5000, tolerance = 0.00001, alpha = 0.1
"""
tmp_file = Path("tmp_input.in")
tmp_file.write_text(input_text)

# Optional context
ctx = SimpleNamespace(input_file=tmp_file, scratch_dir=".")

# Create super dictionary
super_dict = super_dictionary_creator(ctx, export_json=True)

# Inspect dictionary
import json
print(json.dumps(super_dict, indent=2))
```

---
