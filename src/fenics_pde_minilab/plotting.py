"""Matplotlib plotting utilities for generated finite element results."""

from __future__ import annotations

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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
