"""DOLFINx Poisson solve, refinement study, and plots."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fenics_pde_minilab.backend import require_dolfinx
from fenics_pde_minilab.errors import convergence_rates, poisson_exact
from fenics_pde_minilab.plotting import (
    ScalarField,
    scalar_field_from_dolfinx_function,
    save_poisson_convergence,
    save_scalar_field,
)


@dataclass(frozen=True)
class PoissonResult:
    degree: int
    n: int
    h: float
    l2_error: float
    h1_seminorm_error: float


def solve_poisson_case(n: int, degree: int) -> tuple[PoissonResult, object, object]:
    """Solve one Poisson verification case and return metrics plus plot data."""
    require_dolfinx()

    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("These portfolio scripts expect serial execution for plotting.")

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": max(10, 2 * degree + 6)})
    u_exact = ufl.sin(math.pi * x[0]) * ufl.sin(math.pi * x[1])
    forcing = 2.0 * math.pi**2 * u_exact

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    L = forcing * v * dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda points: np.full(points.shape[1], True, dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()

    l2_local = fem.assemble_scalar(fem.form((uh - u_exact) ** 2 * dx))
    h1_local = fem.assemble_scalar(
        fem.form(ufl.dot(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * dx)
    )
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))
    h1_error = math.sqrt(comm.allreduce(h1_local, op=MPI.SUM))

    result = PoissonResult(
        degree=degree,
        n=n,
        h=1.0 / float(n),
        l2_error=l2_error,
        h1_seminorm_error=h1_error,
    )
    return result, domain, uh


def _p1_solution_and_error(domain: object, uh: object) -> tuple[ScalarField, ScalarField]:
    from dolfinx import fem

    V_plot = fem.functionspace(domain, ("Lagrange", 1))
    u_plot = fem.Function(V_plot)
    u_plot.interpolate(uh)
    solution_field = scalar_field_from_dolfinx_function(u_plot)

    exact_values = poisson_exact(solution_field.points[:, 0], solution_field.points[:, 1])
    error_field = ScalarField(
        points=solution_field.points,
        triangles=solution_field.triangles,
        values=np.abs(solution_field.values - exact_values),
    )
    return solution_field, error_field


def _rows_with_rates(results: Iterable[PoissonResult]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    by_degree: dict[int, list[PoissonResult]] = {}
    for result in results:
        by_degree.setdefault(result.degree, []).append(result)

    for degree, degree_results in sorted(by_degree.items()):
        ordered = sorted(degree_results, key=lambda item: item.h, reverse=True)
        h_values = [item.h for item in ordered]
        l2_rates = convergence_rates(h_values, [item.l2_error for item in ordered])
        h1_rates = convergence_rates(h_values, [item.h1_seminorm_error for item in ordered])
        for result, l2_rate, h1_rate in zip(ordered, l2_rates, h1_rates, strict=True):
            rows.append(
                {
                    "degree": float(degree),
                    "n": float(result.n),
                    "h": result.h,
                    "L2_error": result.l2_error,
                    "H1_seminorm_error": result.h1_seminorm_error,
                    "L2_rate": l2_rate,
                    "H1_rate": h1_rate,
                }
            )
    return rows


def write_poisson_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["degree", "n", "h", "L2_error", "H1_seminorm_error", "L2_rate", "H1_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_poisson_convergence(
    mesh_sizes: Iterable[int] = (8, 16, 32, 64),
    degrees: Iterable[int] = (1, 2),
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
) -> list[dict[str, float]]:
    """Run the Poisson refinement study and generate CSV/PNG artifacts."""
    results: list[PoissonResult] = []
    finest_payload: tuple[object, object] | None = None
    mesh_sizes = tuple(mesh_sizes)
    degrees = tuple(degrees)

    for degree in degrees:
        for n in mesh_sizes:
            result, domain, uh = solve_poisson_case(n=n, degree=degree)
            results.append(result)
            if degree == max(degrees) and n == max(mesh_sizes):
                finest_payload = (domain, uh)

    rows = _rows_with_rates(results)
    write_poisson_csv(rows, results_dir / "poisson_convergence.csv")
    save_poisson_convergence(rows, figures_dir / "poisson_convergence.png")

    if finest_payload is not None:
        solution_field, error_field = _p1_solution_and_error(*finest_payload)
        save_scalar_field(solution_field, figures_dir / "poisson_solution.png", "Poisson solution")
        save_scalar_field(error_field, figures_dir / "poisson_error.png", "Poisson pointwise error", cmap="magma")

    return rows
