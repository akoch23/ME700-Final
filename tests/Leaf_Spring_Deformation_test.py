import pytest
import numpy as np
import os

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# Note: To run, imput this command in terminal: mpirun -n 1 pytest -v test_leaf_spring.py

@pytest.fixture(scope="module")
def setup_leaf_spring_model():
    # Geometry and mesh
    num_leaves = 5
    L, W, T, gap = 10.0, 2.0, 0.1, 0.002
    H = num_leaves * T + (num_leaves - 1) * gap
    nx, ny = 50, num_leaves * 5

    domain = mesh.create_rectangle(MPI.COMM_WORLD,
                                   points=np.array([[0.0, 0.0], [L, W]]),
                                   n=np.array([nx, ny]),
                                   cell_type=mesh.CellType.quadrilateral)

    mesh_center_y = (num_leaves - 1) * (T + gap) / 2
    shift_y = mesh_center_y - H / 2
    domain.geometry.x[:, 1] += shift_y

    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))

    # Boundary markers
    def left_marker(x): return np.isclose(x[0], 0.0)
    def right_marker(x): return np.isclose(x[0], L)
    fdim = domain.topology.dim - 1
    left_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, left_marker))
    right_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, right_marker))
    u_bc_val = np.array((0.0, 0.0), dtype=default_scalar_type)
    bcs = [fem.dirichletbc(u_bc_val, left_dofs, V), fem.dirichletbc(u_bc_val, right_dofs, V)]

    # Load application region
    def top_center(x): return np.isclose(x[1], H / 2, atol=1e-5) & (x[0] > L / 3) & (x[0] < 2 * L / 3)
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

    # Material and formulation
    E, nu = default_scalar_type(2.0e5), default_scalar_type(0.3)
    mu = fem.Constant(domain, E / (2 * (1 + nu)))
    lambda_ = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    traction = fem.Constant(domain, np.array([0.0, -1000.0], dtype=default_scalar_type))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(traction, v) * ufl.ds(subdomain_data=facet_tag, subdomain_id=top_center_tag)

    return domain, V, a, L_form, bcs


def test_mesh_dimensions(setup_leaf_spring_model):
    domain, _, _, _, _ = setup_leaf_spring_model
    assert domain.topology.dim == 2
    assert domain.geometry.x.shape[1] == 2


def test_boundary_conditions_applied(setup_leaf_spring_model):
    _, V, _, _, bcs = setup_leaf_spring_model
    assert all(len(bc.dof_indices()) > 0 for bc in bcs)


def test_form_assembly_shapes(setup_leaf_spring_model):
    _, V, a, L_form, bcs = setup_leaf_spring_model
    A = fem.assemble_matrix(fem.form(a), bcs)
    b = fem.assemble_vector(fem.form(L_form))
    assert A.size[0] == A.size[1]
    assert b.getSize() == A.size[0]


def test_solution_is_nonzero(setup_leaf_spring_model):
    domain, V, a, L_form, bcs = setup_leaf_spring_model
    uh = fem.Function(V)
    problem = LinearProblem(a, L_form, bcs=bcs, u=uh,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem.solve()
    assert uh.x.norm() > 0


def test_xdmf_export(setup_leaf_spring_model, tmp_path):
    domain, V, a, L_form, bcs = setup_leaf_spring_model
    uh = fem.Function(V)
    problem = LinearProblem(a, L_form, bcs=bcs, u=uh)
    problem.solve()

    uh.name = "Displacement"
    filename = tmp_path / "test_output.xdmf"
    with io.XDMFFile(domain.comm, str(filename), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)

    assert filename.exists(), "XDMF output file not created"
