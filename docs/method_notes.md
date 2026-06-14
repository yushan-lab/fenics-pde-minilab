# Method Notes

This mini-lab uses manufactured solutions so numerical errors can be measured against known exact fields.

## Poisson

- Geometry: unit square `[0, 1]^2`.
- Mesh: uniform triangular meshes created by DOLFINx.
- Boundary condition: homogeneous Dirichlet values on all boundary facets.
- Weak form: integrate `grad(u) . grad(v)` against `f v`.
- Elements: P1 and P2 Lagrange spaces.
- Error analysis: assembled L2 error and H1 seminorm error against the exact sine solution, using explicit quadrature metadata for the sinusoidal terms.
- Visualization: solution and pointwise error from the finest configured P2 case.

## Heat

- Geometry and boundary conditions match the Poisson problem.
- Initial condition: `sin(pi x) sin(pi y)`.
- Time integrator: backward Euler.
- Default final time: `T = 0.1`.
- Error analysis: final-time L2 error against the exact exponentially decaying sine solution, using explicit quadrature metadata for the sinusoidal exact solution.
- Visualization: initial condition, final numerical solution, final pointwise error, and final-time error trend.

## Environment Limitation

DOLFINx is not a pure Python dependency and is often unavailable in plain Windows Python environments. The repository therefore keeps pure tests independent of DOLFINx and provides a Docker fallback based on `dolfinx/dolfinx:stable`.

The solver setup targets the current DOLFINx `LinearProblem` API by providing deterministic `petsc_options_prefix` values alongside PETSc solver options.
