import pytest
import numpy as np
import os

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type, io
from dolfinx.fem.petsc import LinearProblem
import ufl


@pytest.fixture(scope="module")
def setup_leaf_spring_model():
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

    def left_marker(x): return np.isclose(x[0], 0.0)
    def right_marker(x): return np.isclose(x[0], L)
    fdim = domain.topology.dim - 1
    left_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, left_marker))
    right_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, right_marker))
    u_bc_val = np.array((0.0, 0.0), dtype=default_scalar_type)
    bcs = [fem.dirichletbc(u_bc_val, left_dofs, V), fem.dirichletbc(u_bc_val, right_dofs, V)]

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


def test_leaf_tagging_integrity(setup_leaf_spring_model):
    domain, *_ = setup_leaf_spring_model
    num_leaves = 5
    T, gap = 0.1, 0.002
    L = 10.0
    tdim = domain.topology.dim
    cells = np.arange(domain.topology.index_map(tdim).size_local)
    x = mesh.compute_midpoints(domain, tdim, cells)
    leaf_ids = -1 * np.ones_like(cells, dtype=np.int32)

    for i in range(num_leaves):
        leaf_length = L - i * 1
        y_min = (num_leaves - 1 - i) * (T + gap)
        y_max = y_min + T + (1e-10 if i == num_leaves - 1 else 0.0)
        x_min = (L - leaf_length) / 2
        x_max = x_min + leaf_length
        mask = (x[:, 1] >= y_min) & (x[:, 1] < y_max) & (x[:, 0] >= x_min) & (x[:, 0] <= x_max)
        leaf_ids[cells[mask]] = i

    unique_ids = np.unique(leaf_ids)
    assert len(unique_ids[unique_ids >= 0]) == num_leaves


def test_von_mises_non_negative(setup_leaf_spring_model):
    domain, V, a, L_form, bcs = setup_leaf_spring_model
    uh = fem.Function(V)
    problem = LinearProblem(a, L_form, bcs=bcs, u=uh)
    problem.solve()

    E = 2.0e5
    nu = 0.3
    mu = fem.Constant(domain, E / (2 * (1 + nu)))
    lambda_ = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(2) + 2 * mu * epsilon(u)
    def von_mises(stress):
        s = stress - (1./3) * ufl.tr(stress) * ufl.Identity(2)
        return ufl.sqrt(3./2 * ufl.inner(s, s))

    VM_expr = fem.Expression(von_mises(sigma(uh)), V.element.interpolation_points())
    VM = fem.Function(fem.FunctionSpace(domain, ("DG", 0)))
    VM.interpolate(VM_expr)

    assert np.all(VM.x.array >= 0)
    assert np.max(VM.x.array) > 0


def test_fatigue_life_estimation():
    def estimate_fatigue_life(stress_amplitude, sigma_f_prime, b):
        if stress_amplitude <= 0:
            return None
        return (sigma_f_prime / stress_amplitude) ** (1 / b)

    sigma_f_prime = 650e6
    stress_amplitude = 150e6
    b = -0.1
    N_f = estimate_fatigue_life(stress_amplitude, sigma_f_prime, b)

    assert N_f is not None
    assert N_f > 1e3 and N_f < 1e10
