"""DOLFINx transient heat-equation solve and plots."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fenics_pde_minilab.backend import require_dolfinx
from fenics_pde_minilab.errors import convergence_rates, heat_exact
from fenics_pde_minilab.plotting import (
    ScalarField,
    scalar_field_from_dolfinx_function,
    save_heat_error_trend,
    save_scalar_field,
)


@dataclass(frozen=True)
class HeatResult:
    n: int
    steps: int
    h: float
    dt: float
    theta: float
    final_time: float
    kappa: float
    final_l2_error: float


def solve_heat_case(
    n: int,
    steps: int,
    *,
    degree: int = 1,
    final_time: float = 0.1,
    kappa: float = 1.0,
    theta: float = 0.5,
) -> tuple[HeatResult, object, object, object]:
    """Solve one theta-method heat-equation case."""
    require_dolfinx()

    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("These portfolio scripts expect serial execution for plotting.")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    dt = final_time / float(steps)

    u_old = fem.Function(V)
    u_old.interpolate(lambda x: heat_exact(x[0], x[1], t=0.0, kappa=kappa))
    initial = fem.Function(V)
    initial.x.array[:] = u_old.x.array
    initial.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": max(10, 2 * degree + 6)})
    a = u * v * dx + theta * dt * kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    L = u_old * v * dx - (1.0 - theta) * dt * kappa * ufl.dot(ufl.grad(u_old), ufl.grad(v)) * dx

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
        petsc_options_prefix=f"heat_p{degree}_n{n}_dt{steps}_",
    )
    solution = fem.Function(V)
    for _ in range(steps):
        solution = problem.solve()
        u_old.x.array[:] = solution.x.array
        u_old.x.scatter_forward()

    x = ufl.SpatialCoordinate(domain)
    exact_final = (
        math.exp(-2.0 * math.pi**2 * kappa * final_time)
        * ufl.sin(math.pi * x[0])
        * ufl.sin(math.pi * x[1])
    )
    l2_local = fem.assemble_scalar(fem.form((solution - exact_final) ** 2 * dx))
    l2_error = math.sqrt(comm.allreduce(l2_local, op=MPI.SUM))

    result = HeatResult(
        n=n,
        steps=steps,
        h=1.0 / float(n),
        dt=dt,
        theta=theta,
        final_time=final_time,
        kappa=kappa,
        final_l2_error=l2_error,
    )
    return result, domain, initial, solution


def _p1_fields(domain: object, function: object, *, t: float, kappa: float) -> tuple[ScalarField, ScalarField]:
    from dolfinx import fem

    V_plot = fem.functionspace(domain, ("Lagrange", 1))
    u_plot = fem.Function(V_plot)
    u_plot.interpolate(function)
    solution_field = scalar_field_from_dolfinx_function(u_plot)

    exact_values = heat_exact(solution_field.points[:, 0], solution_field.points[:, 1], t=t, kappa=kappa)
    error_field = ScalarField(
        points=solution_field.points,
        triangles=solution_field.triangles,
        values=np.abs(solution_field.values - exact_values),
    )
    return solution_field, error_field


def _rows(results: Iterable[HeatResult]) -> list[dict[str, float]]:
    ordered = sorted(results, key=lambda item: item.h, reverse=True)
    rates = convergence_rates([result.h for result in ordered], [result.final_l2_error for result in ordered])
    return [
        {
            "n": float(result.n),
            "steps": float(result.steps),
            "h": result.h,
            "dt": result.dt,
            "theta": result.theta,
            "final_time": result.final_time,
            "kappa": result.kappa,
            "final_L2_error": result.final_l2_error,
            "L2_rate": rate,
        }
        for result, rate in zip(ordered, rates, strict=True)
    ]


def write_heat_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n", "steps", "h", "dt", "theta", "final_time", "kappa", "final_L2_error", "L2_rate"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_heat_convergence(
    cases: Iterable[tuple[int, int]] = ((8, 20), (16, 40), (32, 80), (64, 160)),
    *,
    degree: int = 1,
    final_time: float = 0.1,
    kappa: float = 1.0,
    theta: float = 0.5,
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
) -> list[dict[str, float]]:
    """Run the heat refinement study and generate CSV/PNG artifacts."""
    results: list[HeatResult] = []
    finest_payload: tuple[object, object, object] | None = None
    cases = tuple(cases)

    for n, steps in cases:
        result, domain, initial, solution = solve_heat_case(
            n=n,
            steps=steps,
            degree=degree,
            final_time=final_time,
            kappa=kappa,
            theta=theta,
        )
        results.append(result)
        if n == max(case[0] for case in cases):
            finest_payload = (domain, initial, solution)

    rows = _rows(results)
    write_heat_csv(rows, results_dir / "heat_convergence.csv")
    save_heat_error_trend(rows, figures_dir / "heat_error_trend.png")

    if finest_payload is not None:
        domain, initial, solution = finest_payload
        initial_field, _ = _p1_fields(domain, initial, t=0.0, kappa=kappa)
        final_field, error_field = _p1_fields(domain, solution, t=final_time, kappa=kappa)
        save_scalar_field(initial_field, figures_dir / "heat_initial_condition.png", "Heat initial condition")
        save_scalar_field(final_field, figures_dir / "heat_final_solution.png", "Heat final solution")
        save_scalar_field(error_field, figures_dir / "heat_final_error.png", "Heat final pointwise error", cmap="magma")

    return rows
