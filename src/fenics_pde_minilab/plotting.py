"""Matplotlib plotting utilities for generated finite element results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


@dataclass(frozen=True)
class ScalarField:
    points: np.ndarray
    triangles: np.ndarray
    values: np.ndarray


ExactFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def structured_unit_square_triangles(resolution: int) -> np.ndarray:
    """Return triangles for a regular resolution-by-resolution point grid."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    triangles: list[list[int]] = []
    for j in range(resolution - 1):
        for i in range(resolution - 1):
            lower_left = j * resolution + i
            lower_right = lower_left + 1
            upper_left = lower_left + resolution
            upper_right = upper_left + 1
            triangles.append([lower_left, lower_right, upper_right])
            triangles.append([lower_left, upper_right, upper_left])
    return np.asarray(triangles, dtype=int)


def dense_grid_scalar_fields_from_dolfinx_function(
    function: object,
    exact_function: ExactFunction,
    *,
    resolution: int = 257,
    inset: float = 1.0e-12,
) -> tuple[ScalarField, ScalarField, float]:
    """Evaluate a DOLFINx scalar function and exact field on a dense plotting grid."""
    from dolfinx import geometry

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    if not 0.0 <= inset < 0.5:
        raise ValueError("inset must lie in [0, 0.5)")

    mesh = function.function_space.mesh
    coordinates_1d = np.linspace(inset, 1.0 - inset, resolution)
    grid_x, grid_y = np.meshgrid(coordinates_1d, coordinates_1d, indexing="xy")
    points_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    points = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points[:, :2] = points_xy

    tree = geometry.bb_tree(mesh, mesh.topology.dim)
    candidate_cells = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, candidate_cells, points)

    cells = np.empty(points.shape[0], dtype=np.int32)
    missing = 0
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) == 0:
            missing += 1
            cells[i] = -1
        else:
            cells[i] = links[0]

    if missing:
        raise RuntimeError(f"Could not locate containing cells for {missing} dense-grid point(s).")

    numerical_values = np.asarray(function.eval(points, cells), dtype=float).reshape(points.shape[0], -1)[:, 0]
    exact_values = np.asarray(exact_function(points_xy[:, 0], points_xy[:, 1]), dtype=float).reshape(-1)
    error_values = np.abs(numerical_values - exact_values)
    triangles = structured_unit_square_triangles(resolution)

    solution_field = ScalarField(points=points_xy, triangles=triangles, values=numerical_values)
    error_field = ScalarField(points=points_xy, triangles=triangles, values=error_values)
    return solution_field, error_field, float(np.max(error_values))


def scalar_field_from_dolfinx_function(function: object) -> ScalarField:
    """Convert a scalar DOLFINx function on a P1 space to Matplotlib arrays."""
    from dolfinx import plot

    topology, _, geometry = plot.vtk_mesh(function.function_space)
    topology = np.asarray(topology)
    if topology.ndim == 1:
        nodes_per_cell = int(topology[0])
        triangles = topology.reshape((-1, nodes_per_cell + 1))[:, 1:]
    else:
        triangles = topology

    if triangles.shape[1] != 3:
        triangles = triangles[:, :3]

    values = np.asarray(function.x.array, dtype=float)
    return ScalarField(
        points=np.asarray(geometry[:, :2], dtype=float),
        triangles=np.asarray(triangles, dtype=int),
        values=values,
    )


def save_scalar_field(field: ScalarField, path: Path, title: str, cmap: str = "viridis") -> None:
    ensure_parent(path)
    triangulation = mtri.Triangulation(field.points[:, 0], field.points[:, 1], field.triangles)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    contour = ax.tricontourf(triangulation, field.values, levels=32, cmap=cmap)
    ax.triplot(triangulation, color="black", linewidth=0.15, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_poisson_convergence(rows: Iterable[dict[str, float]], path: Path) -> None:
    ensure_parent(path)
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["degree"]), []).append(row)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for degree, degree_rows in sorted(grouped.items()):
        ordered = sorted(degree_rows, key=lambda item: item["h"], reverse=True)
        h = [item["h"] for item in ordered]
        l2 = [item["L2_error"] for item in ordered]
        h1 = [item["H1_seminorm_error"] for item in ordered]
        ax.loglog(h, l2, marker="o", label=f"P{degree} L2")
        ax.loglog(h, h1, marker="s", linestyle="--", label=f"P{degree} H1 seminorm")

    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error")
    ax.set_title("Poisson convergence")
    ax.grid(True, which="both", linewidth=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heat_error_trend(rows: Iterable[dict[str, float]], path: Path) -> None:
    ensure_parent(path)
    ordered = sorted(rows, key=lambda item: item["h"], reverse=True)
    h = [item["h"] for item in ordered]
    error = [item["final_L2_error"] for item in ordered]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.loglog(h, error, marker="o")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("final-time L2 error")
    ax.set_title("Heat-equation final-time error")
    ax.grid(True, which="both", linewidth=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)
