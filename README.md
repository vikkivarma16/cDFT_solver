# cDFT_solver

`cDFT_solver` is a Python package for **classical Density Functional Theory (cDFT)** and related liquid-state calculations. It is designed around a modular workflow in which interactions, structural data, reference systems, free-energy models, and thermodynamic calculations can be assembled and reused independently.

The package supports both direct model-defined calculations and workflows that incorporate externally supplied information such as tabulated pair potentials or radial distribution functions.

**Author:** Vikki Anand Varma  
**Email:** vikkivarma16@gmail.com

> **Looking for full documentation?**  
> This README is intended as a quick orientation and starting point. Detailed input-file syntax, data contracts, module-level APIs, custom closures, supplied-data workflows, inversion methods, thermodynamic calculations, and complete worked examples are documented in the accompanying **cDFT Solver User and Developer Manual**.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/vikkivarma16/cDFT_solver.git
```

### Main requirements

- Python ≥ 3.7
- NumPy
- SciPy
- the standard scientific Python dependencies used by the package

For a complete description of installation assumptions and package organization, see **Part I of the manual**.

---

## What the package does

At a high level, `cDFT_solver` provides tools for:

- structured input parsing,
- one- and multicomponent pair interactions,
- hard-core/reference and mean-field decompositions,
- isotropic radial distribution functions,
- Ornstein-Zernike based structural calculations,
- direct correlation functions,
- structure factors,
- second-virial and integrated interaction quantities,
- free-energy construction,
- equations of state,
- coexistence and binodal calculations,
- effective-potential inversion,
- supplied RDF and supplied-potential workflows,
- custom analytical potentials,
- user-defined closure relations,
- multi-state inversion using structural data from several thermodynamic state points.

The package is intentionally modular: the output of one stage can usually be inspected, modified, stored, and reused as the input to another stage.

---

# Quick mental model

A typical calculation follows the pattern

```text
input file / Python dictionary / supplied data
                    │
                    ▼
          system configuration
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   pair interactions      supplied data
          │                   │
          └─────────┬─────────┘
                    ▼
          structural / reference
              calculations
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         RDF      c(r)       S(k)
          │
          ▼
 interaction strengths / free energy
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
         EOS              coexistence
```

Not every workflow uses every stage. For example, a potential-inversion calculation may begin from supplied RDF data, whereas an ordinary EOS calculation may begin entirely from analytical pair interactions.

The full data-flow diagrams and input/output contracts are given in **Parts II–VI of the manual**.

---

# Minimal input example

The package uses a structured text input that is converted into a nested Python dictionary.

A simple interaction definition may look like

```text
species = a, b

interaction primary: aa type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.0
interaction primary: ab type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.5
interaction primary: bb type = gs, sigma = 1.414, cutoff = 3.5, epsilon = 2.0
```

Parse it with

```python
from cdft_solver.generators.parameters.advance_dictionary import (
    super_dictionary_creator,
)

system = super_dictionary_creator(
    input_file="system.in",
    export_json=False,
)
```

The result is a nested dictionary that can be passed to the calculation modules.

The input language is substantially richer than this small example. The manual documents:

- all major input blocks,
- accepted values,
- required and optional fields,
- pair naming,
- density and temperature specification,
- RDF settings,
- closure selection,
- free-energy settings,
- supplied data,
- tabulated interactions,
- numerical controls,
- and the exact parsed dictionary structure.

See **Part II — Input Language, Configuration, and Data Contracts**.

---

# Pair interactions

Pair interactions may be defined in several ways.

## Built-in analytical potentials

A pair can be specified directly in the input dictionary using one of the implemented potential models.

Conceptually,

```python
pair_definition = {
    "type": "...",
    "sigma": ...,
    "epsilon": ...,
    "cutoff": ...
}
```

can be converted into a vectorized callable

\[
U_{ij}(r).
\]

## Potential supplied from a file

A pair interaction can also be supplied numerically using tabulated radial data. This allows a potential generated by simulation, inversion, another theory, or an external program to be reused by `cDFT_solver`.

## Runtime-defined analytical potential

Advanced workflows can register a Python-defined potential at runtime, allowing a new functional form to be used without permanently adding it to the core package.

The exact file format, registration interface, interpolation behavior, and compatibility with downstream modules are documented in **Part VI of the manual**.

---

# Interaction decomposition

For workflows that use a reference/mean-field decomposition, the package provides utilities for constructing:

- the **raw interaction**,
- the **hard-core/reference contribution**,
- the **mean-field contribution**, and
- the corresponding total numerical potential.

Conceptually,

\[
U_{ij}^{\mathrm{raw}}(r)
=
\sum_{\ell} U_{ij}^{(\ell)}(r),
\]

while reference-based calculations can use a decomposition of the form

\[
U_{ij}(r)
=
U_{ij}^{\mathrm{ref}}(r)
+
U_{ij}^{\mathrm{MF}}(r).
\]

The exact mapping depends on the chosen interaction model and workflow.

For implementation details and return dictionaries, see the **Potential and Reference-System chapters in the manual**.

---

# Structural calculations

The package supports isotropic structural calculations for one- and multicomponent systems.

Depending on the workflow, the solver can determine or analyze quantities such as

\[
g_{ij}(r),
\qquad
h_{ij}(r)=g_{ij}(r)-1,
\qquad
c_{ij}(r),
\qquad
S_{ij}(k).
\]

Closures can be chosen pair by pair.

In addition to built-in closures, a closure can be supplied manually as a Python callable. This is useful for testing new liquid-state approximations without modifying the core closure dispatcher.

For the required callable signature, array conventions, pair-specific assignment, and examples, see **Part VI — Advanced Extensibility, Supplied Data, and Multistate Inversion**.

---

# Supplied structural data

`cDFT_solver` can combine calculated interactions with externally supplied structural information.

For example, a multicomponent calculation may use:

- analytical potentials for some pairs,
- a tabulated potential for another pair,
- a supplied RDF for only one selected interaction,
- closure-computed RDFs for the remaining pairs.

This makes it possible to construct hybrid workflows in which only the information that is actually known externally needs to be supplied.

Because different consumers use slightly different internal field names for radial data, the complete manual includes a compatibility table for the `x/y`, `r/g`, and `r/U` data schemas.

See **Part VI** before building a mixed supplied-data workflow.

---

# Effective-potential inversion

The package contains workflows for reconstructing effective pair interactions from structural information.

A typical inversion workflow is

```text
supplied RDF data
       │
       ▼
initial effective potential
       │
       ▼
OZ / closure calculation
       │
       ▼
comparison with target RDF
       │
       ▼
potential update
       │
       └────── iterate until convergence
```

The inversion machinery supports more than a single RDF at a single state point.

It can be used for:

- one-component inversion,
- multicomponent inversion,
- inversion of only selected pairs,
- keeping known pair potentials fixed while inferring unknown ones,
- and fitting a shared effective interaction against RDF data from several thermodynamic state points.

The detailed inversion algorithm, data layout, state-point organization, update rules, convergence controls, masks, and exported potentials are documented in the manual.

---

# Thermodynamic workflows

Once the interaction and reference information is available, the package can be used to construct thermodynamic quantities such as

- free-energy density,
- chemical potentials,
- pressure,
- equations of state,
- integrated interaction strengths,
- and phase coexistence.

A generalized calculation chain is

```text
interaction model
      │
      ▼
reference contribution
      +
interaction correction
      │
      ▼
free energy
      │
 ┌────┴────┐
 ▼         ▼
EOS     coexistence
```

Worked one-component and binary EOS calculations, integrated-strength workflows, and binodal calculations are given in **Part III of the manual**.

---

# Portable workflow philosophy

The example scripts distributed with the package demonstrate particular calculations, but their folder names are not part of the API.

A portable user script should normally define paths explicitly, for example

```python
from pathlib import Path

input_file = Path("inputs/system.in")
rdf_file = Path("data/rdf_ab.dat")
output_dir = Path("results")
```

and then pass those paths to the relevant parser or calculation function.

The manual rewrites the bundled examples in this general form so that users are not required to reproduce the repository's example-directory structure.

---

# Where to look in the manual

The README intentionally avoids reproducing the full reference documentation.

| Manual part | Use it when you need... |
|---|---|
| **Part I** | installation, orientation, and package architecture |
| **Part II** | complete `.in` syntax, parsed dictionaries, data formats, and input/output contracts |
| **Part III** | EOS, RDF, structural analysis, free energy, integrated strengths, and coexistence workflows |
| **Part IV** | audited example inputs, validation, and input-design guidance |
| **Part V** | module reference, generated outputs, and package-maintenance notes |
| **Part VI** | custom closures, external potentials, supplied RDFs, mixed-data calculations, and multistate inversion |

If a field, function argument, array shape, output dictionary, or advanced workflow is unclear, the manual should be treated as the authoritative companion to this README.

---

# Recommended starting workflow

For a first calculation:

1. Install the package.
2. Create a small `.in` file defining the species and pair interactions.
3. Parse it with `super_dictionary_creator`.
4. Verify the resulting dictionary.
5. Run the desired structural or thermodynamic module.
6. Inspect the returned data before enabling file export.
7. Move to supplied-data, custom-closure, or inversion workflows only after the basic model-defined calculation is working.

This keeps the first calculation transparent and makes it easier to distinguish input-definition problems from numerical-convergence problems.

---

# Documentation philosophy

The package documentation is split deliberately into two layers:

### README

Use this file to answer:

- What is `cDFT_solver`?
- What can it calculate?
- How do I install it?
- How is a calculation organized?
- What is the smallest useful example?
- Where should I look next?

### User and Developer Manual

Use the manual to answer:

- What exactly does this input keyword mean?
- What shape should this array have?
- Which module consumes this data?
- What does this function return?
- How do I supply an RDF or a potential?
- How do I write a custom closure?
- How are several data sources combined?
- How does multistate inversion work?
- How do I reproduce a complete EOS or coexistence workflow?
- What are the current implementation caveats?

This separation keeps the README readable while allowing the manual to remain technically rigorous.

---

## Citation and contact

If you use `cDFT_solver` in scientific work, please cite the relevant methodological publications associated with the calculation performed.

For questions about the package:

**Vikki Anand Varma**  
**Email:** vikkivarma16@gmail.com
