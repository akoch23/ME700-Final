# Leaf Spring Simulation: ME 700 Applied Skills

## Technical Knowledge
- ### Matrix structural analysis using FEM
  The code formulates and solves a linear elasticity problem using finite element methods (FEM), incorporating stress-strain    relationships and Lamé parameters to represent mechanical behavior. 
- ### Finite element analysis implementation
  Quadrilateral meshing, Dirichlet boundary conditions, tagged subdomains, and the use of ufl for defining weak forms.

## Software Development & Implementation
- ### Python code development for computational mechanics
  The code is written entirely in Python using modern scientific libraries such as NumPy, DOLFINx, UFL, PyVista
- ### Debugging and Runtime Validation
  The script includes error handling and HDF5 content inspection to verify data output integrity
- ### Basic Documentation and Code Structuring
  Comments describing various variables, functions, etc. organized code sections (e.g., geometry, material, boundary conditions, solution), and output instructions
  
## Integration & Application
- ### Integration of theory and computation
  The code integrates core mechanical concepts (elastic behavior, load application, and geometric layering) with computational implementation using DOLFINx
- ### Reusable Design
  Mesh generation, tagging, material setup, and loading are written in a general, parametric way (e.g., configurable number of leaves)
