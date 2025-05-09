# ME700-Final

Final Project Description:
This project simulates the deformation and stress behavior of a stacked multi-leaf spring using the finite element method in [DOLFINx](https://github.com/FEniCS/dolfinx). The simulation includes:

- Linear elasticity
- Contact formulation between adjacent leaves using a penalty method
- von Mises stress analysis
- Fatigue life estimation
- Animated deformation visualization using PyVista
- XDMF/HDF5 file exports for visualization in ParaView

---

## Requirements

Ensure the following packages are installed in your environment:

- Python 3.10+
- dolfinx
- mpi4py
- numpy
- ufl
- pyvista
- h5py *(optional, for inspecting HDF5 files)*
- Xvfb *(only required for headless servers)*

You can install these using conda (recommended with DOLFINx):

```bash
conda create -n dolfinx310 -c conda-forge dolfinx pyvista h5py
conda activate dolfinx310
```

# Note: Code Failure
During final testing, my fenicsx environment suffered some kind of error regarding PETSc - I was unable to even run previous working fenicsx code, even from the class course repository. I am unsure of how to fix this issue and looking up solutions online and via AI has yielded nothing useful.
