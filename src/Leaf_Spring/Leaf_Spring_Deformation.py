# Import Necessary Libraries
import os
import numpy as np
import dolfinx
from dolfinx import io, mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
import pyvista as pv
import h5py


# Start X Virtual Framebuffer (necessary for headless rendering)
pv.start_xvfb()

# Number of leaves in the stack
num_leaves = 5

# Define the geometry of each leaf (e.g., initial length, width, thickness)
L = 10.0  # Initial length of the top leaf
W = 2  # Width of each leaf
T = 0.1  # Thickness of each leaf
gap = 0.002  # Space between leaves (for modeling)
H = num_leaves * T + ((num_leaves - 1) * gap)  # Total height of stack

# Define the length decrement for each successive leaf (can be a fixed amount or percentage)
length_decrement = 1  # Length reduction per leaf, for example 0.5 units per leaf

# Mesh resolution
nx = 50
ny = num_leaves * 5  # Increase vertical resolution for smoothness

# Create single stacked domain as a rectangle
domain = mesh.create_rectangle(MPI.COMM_WORLD, points=np.array([[0.0, 0.0], [L, W]]), n=np.array([nx, ny]), cell_type=mesh.CellType.quadrilateral)

# Calculate the center of the mesh
mesh_center_y = (num_leaves - 1) * (T + gap) / 2
# Shift the mesh to the middle of the grid (center it vertically)
shift_y = mesh_center_y - H / 2
# Apply vertical shift to the entire mesh
domain.geometry.x[:, 1] += shift_y

# Function Space
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))


# Boundary Conditions
# Define marker function for left and right boundaries
def left_marker(x):
    return np.isclose(x[0], 0.0)  # Left boundary is at x=0

def right_marker(x):
    return np.isclose(x[0], L)  # Right boundary is at x=L

fdim = domain.topology.dim - 1
# Locate degrees of freedom for left and right boundaries
left_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, left_marker))
right_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, right_marker))

# Apply boundary conditions
u_bc_val = np.array((0.0, 0.0), dtype=default_scalar_type)
bcs = [fem.dirichletbc(u_bc_val, left_dofs, V), fem.dirichletbc(u_bc_val, right_dofs, V)]

# Load application
H_center = H / 2
def top_center(x):
    return np.isclose(x[1], H_center, atol=1e-5) & (x[0] > L / 3) & (x[0] < 2 * L / 3)
top_center_facets = mesh.locate_entities_boundary(domain, fdim, top_center)
top_center_tag = 4
left_facets = mesh.locate_entities_boundary(domain, fdim, left_marker)
right_facets = mesh.locate_entities_boundary(domain, fdim, right_marker)
marked_facets = np.hstack([left_facets, right_facets, top_center_facets])
marked_values = np.hstack([
    np.full_like(left_facets, 1),
    np.full_like(right_facets, 2),
    np.full_like(top_center_facets, top_center_tag)
])
sort_idx = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sort_idx], marked_values[sort_idx])

# Leaf tagging
tdim = domain.topology.dim
cells = np.arange(domain.topology.index_map(tdim).size_local)
x = mesh.compute_midpoints(domain, tdim, cells)
# Default to -1 (untagged)
leaf_ids = -1 * np.ones_like(cells, dtype=np.int32)

for i in range(num_leaves):
    leaf_length = L - i * length_decrement
    y_min = (num_leaves - 1 - i) * (T + gap)
    y_max = y_min + T + (1e-10 if i == num_leaves - 1 else 0.0)
    x_min = (L - leaf_length) / 2
    x_max = x_min + leaf_length
    mask = (x[:, 1] >= y_min) & (x[:, 1] < y_max) & (x[:, 0] >= x_min) & (x[:, 0] <= x_max)
    leaf_ids[cells[mask]] = i  # Assign leaf index

leaf_tag = mesh.meshtags(domain, tdim, cells, leaf_ids)
print("All leaves tagged.")

# Material properties
E = default_scalar_type(2.0e5)
nu = default_scalar_type(0.3)
# Lamé parameters (E, nu)
mu = fem.Constant(domain, E / (2 * (1 + nu)))  # Shear modulus
lambda_ = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))  # First Lamé parameter
# Penalty parameter for contact
penalty = fem.Constant(domain, 1e7)  # Adjust for mesh density / convergence

# Load
F = 1000.0
traction = fem.Constant(domain, np.array([0.0, -F], dtype=default_scalar_type))

# Variational formulation (linear elasticity)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Strain tensor (symmetric gradient)

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  # Stress tensor

def von_mises(stress): 
    s = stress - (1./3) * ufl.tr(stress) * ufl.Identity(2)
    return ufl.sqrt(3./2 * ufl.inner(s, s))

# Define Variational Problem 
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

# Contact formulation
def contact_facets_between(leaf_i, leaf_j):
    tdim = domain.topology.dim
    fdim = tdim - 1
    cell_map = leaf_tag.indices
    cell_values = leaf_tag.values
    cells_i = cell_map[cell_values == leaf_i]
    cells_j = cell_map[cell_values == leaf_j]
    domain.topology.create_connectivity(tdim, fdim)
    facets_i = mesh.exterior_facet_indices(domain, cells_i)
    facets_j = mesh.exterior_facet_indices(domain, cells_j)
    mids_i = mesh.compute_midpoints(domain, fdim, facets_i)
    mids_j = mesh.compute_midpoints(domain, fdim, facets_j)
    contact_facets = []
    for f_i, mid_i in zip(facets_i, mids_i):
        for mid_j in mids_j:
            if np.allclose(mid_i, mid_j, atol=1e-8):
                contact_facets.append(f_i)
                break
    return np.array(contact_facets, dtype=np.int32)
    
# Add contact terms for each pair of adjacent leaves
for i in range(num_leaves - 1):
    contact_facets = contact_facets_between(i, i + 1)
    if len(contact_facets) == 0:
        continue

    # Tag contact interface
    contact_marker = i + 100
    contact_tag = mesh.meshtags(domain, fdim, contact_facets,
                                np.full(len(contact_facets), contact_marker, dtype=np.int32))

    # Contact term: penalty * jump in normal direction
    n = ufl.FacetNormal(domain)
    jump = ufl.dot(u, n) * ufl.dot(v, n)
    a += penalty * jump * ufl.ds(subdomain_data=contact_tag, subdomain_id=contact_marker)


L_form = ufl.dot(traction, v) * ufl.ds(subdomain_data=facet_tag, subdomain_id=top_center_tag)
uh = fem.Function(V)
problem = LinearProblem(a, L_form, bcs=bcs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()
print("Problem Solved!")

# Format for Paraview output
uh_array = uh.x.array.reshape((domain.geometry.x.shape[0], -1))
uh_array = uh_array[:, :2]  # ensure 2D
uh.x.array[:] = uh_array.flatten()
uh.name = "Displacement"

# Compute von Mises stress
stress_expr = fem.Expression(sigma(uh), V.element.interpolation_points())
stress_tensor = fem.Function(fem.TensorFunctionSpace(domain, ("DG", 0)))
stress_tensor.interpolate(stress_expr)

VM_expr = fem.Expression(von_mises(sigma(uh)), V.element.interpolation_points())
VM = fem.Function(fem.FunctionSpace(domain, ("DG", 0)))
VM.name = "von_Mises_stress"
VM.interpolate(VM_expr)

VM_values = VM.x.array
max_vm_stress = np.max(VM_values)
print(f"Maximum von Mises stress (FEM): {max_vm_stress:.2f} Pa")

# Evaluate von Mises stress at bottom midspan
bottom_mid = np.array([[L / 2, 0.0]])  # x = midspan, y = bottom leaf
bbox_tree = BoundingBoxTree(domain, domain.topology.dim)
cells = compute_collisions_points(bbox_tree, bottom_mid.T, domain)
if len(cells[0]) > 0:
    cell = cells[0][0]
    vm_local = VM.eval(bottom_mid[0], cell)
    print(f"von Mises at bottom midspan: {vm_local[0]:.2f} Pa")
else:
    vm_local = [None]
    print("Could not evaluate von Mises at bottom midspan.")

# FEM-Computed vs Analytical Solution
def compare_fem_vs_analytical(F, L, E, N, b, h, fem_displacements):
    p = 1.5  # geometric factor for simple support
    q = 1 / 6  # geometric factor for simple support

    sigma_analytical = (p * F * L) / (N * b * h**2)
    delta_analytical = (q * F * L**3) / (E * N * b * h**3)

    max_disp_y = np.min(fem_displacements[:, 1])  # downward max

    print("\\n--- FEM vs Analytical ---")
    print(f"Analytical deflection: {delta_analytical:.6f} m")
    print(f"FEM deflection:        {abs(max_disp_y):.6f} m")
    print(f"% Error:               {100 * abs((abs(max_disp_y) - delta_analytical) / delta_analytical):.2f}%")
    print(f"Analytical stress:     {sigma_analytical:.2f} Pa\\n")

def estimate_fatigue_life(stress_amplitude, sigma_f_prime, b, label=""):
    if stress_amplitude <= 0:
        print(f"{label}Invalid stress amplitude for fatigue calculation.")
        return None
    N_f = (sigma_f_prime / stress_amplitude) ** (1 / b)
    print(f"{label}Estimated fatigue life: {N_f:.2e} cycles")
    return N_f

displacements = uh.x.array.reshape((domain.geometry.x.shape[0], 2))
compare_fem_vs_analytical(F, L, E, num_leaves, W, T, displacements)
stress_analytical = (1.5 * F * L) / (num_leaves * W * T**2)
sigma_f_prime = 650e6  # Pa
b = -0.1
print("Fatigue Estimates:")
estimate_fatigue_life(stress_analytical, sigma_f_prime, b, label="[Analytical] ")
estimate_fatigue_life(max_vm_stress, sigma_f_prime, b, label="[FEM Max VM] ")
if vm_local[0] is not None:
    estimate_fatigue_life(vm_local[0], sigma_f_prime, b, label="[FEM Mid-Span] ")

# Create XDMF file to save the mesh and solution
with io.XDMFFile(domain.comm, "leaf_spring.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(VM)
print("Mesh and solution saved in XDMF format.")

# Export Leaf Tags
with io.XDMFFile(domain.comm, "leaf_tag.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(leaf_tag, domain.geometry)
print("Leaf tag meshtags exported to 'leaf_tag.xdmf'")
print(" Simulation with contact complete. You may use ParaView to visualize displacement and leaf tags.")


# Set up VTK mesh for visualization
topology, cells, geometry = plot.vtk_mesh(uh.function_space)
grid = pv.UnstructuredGrid(topology, cells, geometry)

# Add leaf ID to grid
grid.cell_data["leaf_id"] = leaf_ids

# Plotting Figure
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("leaf_spring_deformation.gif", fps=10)

# Initialize deformation field
grid["u"] = uh.x.array.reshape(geometry.shape[0], 2)
grid.set_active_vectors("u")
warped = grid.warp_by_vector("u", factor=10.0)
plotter.add_mesh(warped, show_edges=True, lighting=False, cmap="viridis")
plotter.view_xy()


# Time-stepping loop: increase load in steps
for step in range(1, 11):
    scale = step / 10.0
    traction.value[:] = np.array([0.0, -F * scale], dtype=default_scalar_type)

    # Reassemble RHS with updated load
    L_form = ufl.dot(traction, v) * ufl.ds(subdomain_data=facet_tag, subdomain_id=top_center_tag)
    problem = LinearProblem(a, L_form, bcs=bcs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()
    uh.x.scatter_forward()

    # Update displacement in grid and warp
    grid.point_data["u"][:] = uh.x.array.reshape((domain.geometry.x.shape[0], 2))
    warped.points[:] = grid.warp_by_vector("u", factor=10.0).points

    plotter.write_frame()

plotter.close()
print(\"Animation saved as 'leaf_spring_deformation.gif'\")

'''
# HDF5 inspection
try:
    with h5py.File("leaf_spring.h5", "r") as f:
        print("Inspecting HDF5 file contents (leaf_spring.h5):")
        def print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  - {name}: shape = {obj.shape}")
        f.visititems(print_tree)
        if "Function/Displacement/0" in f:
            print("Displacement' dataset found.")
        else:
            print("Displacement' dataset NOT found in HDF5 file.")
except Exception as e:
    print("HDF5 inspection failed:", e)

print("Simulation complete. To visualize:")
print("- Open 'leaf_spring.xdmf' in ParaView and use 'Warp By Vector' on 'Displacement'")
print("- Open 'leaf_tag.xdmf' to color mesh by leaf ID")
'''
