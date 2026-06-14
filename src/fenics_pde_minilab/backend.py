"""Small runtime guard for optional DOLFINx dependencies."""

from __future__ import annotations


DOLFINX_INSTALL_MESSAGE = """DOLFINx is required for the solver scripts but is not available in this Python environment.

Recommended local options:
1. Use the Docker fallback: make docker-reproduce
2. Run inside an existing DOLFINx/FEniCSx environment, then execute: make reproduce

The pure manufactured-solution tests do not require DOLFINx.
"""


def require_dolfinx() -> None:
    """Raise a readable error if DOLFINx is unavailable."""
    try:
        import dolfinx  # noqa: F401
        import ufl  # noqa: F401
        from petsc4py import PETSc  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(DOLFINX_INSTALL_MESSAGE) from exc
