# Stacked Leaf Spring Simulation (DOLFINx)

This project simulates a 2D stacked leaf spring using the finite element method (FEM) in DOLFINx. It models linear elastic behavior under a distributed load, includes per-leaf tagging, and supports visualization in ParaView.

## Overview

The simulation:
- Creates a rectangular domain representing a stack of leaf springs.
- Applies vertical load to the center of the top surface.
- Solves a linear elasticity problem using FEM.
- Tags mesh cells according to which leaf they belong to.
- Exports results for visualization in ParaView.

---

## Key Parameters

| Parameter          | Description                              | Value / Formula               |
|-------------------|------------------------------------------|-------------------------------|
| `num_leaves`       | Number of leaf springs in the stack      | 5                             |
| `L`, `W`, `T`       | Length, width, and thickness of leaves   | `L=10.0`, `W=2`, `T=0.1`      |
| `gap`              | Space between leaves                     | 0.002                         |
| `length_decrement` | Reduction in length per leaf             | 1.0 unit                      |
| `nx`, `ny`         | Mesh resolution in x/y                   | 50, `5 × num_leaves`          |
| `F`                | Load magnitude                           | 1000 N/m²                     |

---

## Geometry and Meshing

- A single rectangular domain is created with a height covering all leaves plus gaps.
- The vertical coordinate is shifted so that the leaves are centered around `y = 0`.
- Mesh is structured with quadrilateral elements.
- Cells are tagged to associate each region with a specific leaf.

---

## Variational Formulation

Linear elasticity is used:

\[
\sigma(u) = \lambda \nabla \cdot u \cdot I + 2\mu \cdot \varepsilon(u), \quad \varepsilon(u) = \text{sym}(\nabla u)
\]

where:
- \( \mu = \frac{E}{2(1 + \nu)} \) — shear modulus
- \( \lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)} \) — Lamé parameter

The weak form is:

\[
\int_\Omega \sigma(u) : \varepsilon(v) \, dx = \int_{\Gamma_N} t \cdot v \, ds
\]

---

## Boundary and Loading Conditions

- **Dirichlet BCs**: Zero displacement on the left and right edges.
- **Neumann BCs**: Downward traction applied at the top-center region of the stack.
- **Tagged boundaries** are used for selective loading and visualization.

---

## Output and Visualization

- **Mesh and solution** are saved to `leaf_spring.xdmf` (for ParaView).
- **Leaf tags** are exported to `leaf_tag.xdmf`.
- **HDF5 output** is checked programmatically for integrity using `h5py`.

### ParaView Instructions:
1. Open `leaf_spring.xdmf` and use `Warp By Vector` on `Displacement`.
2. Open `leaf_tag.xdmf` to color by leaf ID.

---

## Optional Contact Modeling (Commented Section)

There is a commented-out scaffold for implementing a contact penalty method between adjacent leaves. It defines a weak contact formulation based on enforcing displacement continuity at shared interfaces.


## File Outputs

| File                | Description                                     |
|---------------------|-------------------------------------------------|
| `leaf_spring.xdmf`  | Contains mesh and displacement field            |
| `leaf_tag.xdmf`     | Tagged leaf IDs for visualization               |
| `leaf_spring.h5`    | HDF5 backend file (may be inspected with h5py)  |
