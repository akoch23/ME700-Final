# Import Necessary Libraries
import os
import numpy as np
import dolfinx
from dolfinx import io, mesh, fem, default_scalar_type, la, plot
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

# --- Function Space ---
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))


# --- Boundary Conditions ---
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

# Load Application region
def top_center(x):
    return np.isclose(x[1], H / 2, atol=1e-5) & (x[0] > L / 3) & (x[0] < 2 * L / 3)

top_center_facets = mesh.locate_entities_boundary(domain, fdim, top_center)
top_center_tag = 4

# Mark all facets
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

# --- Leaf tagging ---
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

# Variational formulation (linear elasticity)
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Strain tensor (symmetric gradient)

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)  # Stress tensor

# Define Variational Problem 
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# Load Vector
F = 1000.0  # N/m², adjust to suit physical realism
traction = fem.Constant(domain, np.array([0.0, -F], dtype=default_scalar_type))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(traction, v) * ufl.ds(subdomain_data=facet_tag, subdomain_id=top_center_tag)


# Solve the linear variational problem
uh = fem.Function(V)
problem = LinearProblem(a, L_form, bcs=bcs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()
print("Problem Solved!")

# Strip any extra dimensions for ParaView compatibility
uh_array = uh.x.array.reshape((domain.geometry.x.shape[0], -1))
uh_array = uh_array[:, :2]  # ensure 2D
uh.x.array[:] = uh_array.flatten()
uh.name = "Displacement"

# Create XDMF file to save the mesh and solution
with io.XDMFFile(domain.comm, "leaf_spring.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
print("Mesh and solution saved in XDMF format.")

# --- Export Leaf Tags ---
with io.XDMFFile(domain.comm, "leaf_tag.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(leaf_tag, domain.geometry)
print("Leaf tag meshtags exported to 'leaf_tag.xdmf'")

# HDF5 inspection
try:
    with h5py.File("leaf_spring.h5", "r") as f:
        print("Inspecting HDF5 file contents (leaf_spring.h5):")
        def print_tree(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  - {name}: shape = {obj.shape}")
        f.visititems(print_tree)
        if "Function/Displacement/0" in f:
            print("✅ 'Displacement' dataset found.")
        else:
            print("⚠️ 'Displacement' dataset NOT found in HDF5 file.")
except Exception as e:
    print("⚠️ HDF5 inspection failed:", e)

print("✅ Simulation complete. To visualize:")
print("- Open 'leaf_spring.xdmf' in ParaView and use 'Warp By Vector' on 'Displacement'")
print("- Open 'leaf_tag.xdmf' to color mesh by leaf ID")





'''
Leaf-Leaf Interaction
'''
'''
# Define contact interaction (simplified example for two leaves)
# Let's assume the contact only exists between the top and bottom of two adjacent leaves

# Define test and trial functions
u1 = dolfinx.fem.Function(V)  # Displacement on the first leaf
u2 = dolfinx.fem.Function(V)  # Displacement on the second leaf
v1 = ufl.TestFunction(V)  # Test function for first leaf
v2 = ufl.TestFunction(V)  # Test function for second leaf

# Contact term: enforce displacement equality at the interface
contact_term = ufl.inner(u1 - u2, v1) * ufl.ds  # Penalty method

# Add this to the weak form of the problem
a_contact = contact_term  # Apply contact force as a weak formulation

# Assemble the system
L_vec_contact = dolfinx.fem.assemble_vector(a_contact)
'''


'''
Solving
'''
'''
# Assemble the system matrices
A = dolfinx.fem.assemble_matrix(a, [bc])
L_vec = dolfinx.fem.assemble_vector(L)
L_vec_contact = dolfinx.fem.assemble_vector(a_contact)

# Combine the load vectors
L_vec_total = L_vec + L_vec_contact

# Apply boundary conditions to the load vector
dolfinx.fem.apply_dirichlet_bc(L_vec_total, [bc])

# Solve the system
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setType('preonly')  # Direct solver
solver.setTolerances(rtol=1e-8)
solver.solve(L_vec_total, u.vector)
'''

