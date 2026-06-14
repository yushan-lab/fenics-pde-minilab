# fenics-pde-minilab

`fenics-pde-minilab` is a compact portfolio project for physical simulation and AI for Science research-assistant work. It demonstrates a conventional finite element workflow with manufactured solutions: geometry definition, mesh generation, boundary conditions, weak forms, linear solves, mesh refinement, error analysis, post-processing, and visualization.

The implementation targets FEniCSx / DOLFINx. The current Codex environment may not provide DOLFINx on Windows; when that happens, the solver scripts fail with a clear installation message and the numerical results below remain pending rather than invented.

## Problems

Poisson equation on the unit square:

```text
-Delta u = f        in Omega = [0, 1]^2
u = 0               on boundary(Omega)
u_exact(x, y) = sin(pi x) sin(pi y)
f(x, y) = 2 pi^2 sin(pi x) sin(pi y)
```

Weak form: find `u` in `H_0^1(Omega)` such that

```text
integral_Omega grad(u) . grad(v) dx = integral_Omega f v dx
```

for all test functions `v` in `H_0^1(Omega)`.

Heat equation on the unit square:

```text
u_t - kappa Delta u = 0
u = 0 on boundary(Omega)
u(x, y, 0) = sin(pi x) sin(pi y)
u_exact(x, y, t) = exp(-2 pi^2 kappa t) sin(pi x) sin(pi y)
```

The transient solve uses backward Euler:

```text
integral_Omega u^{m+1} v dx
+ dt kappa integral_Omega grad(u^{m+1}) . grad(v) dx
= integral_Omega u^m v dx
```

## Refinement And Error Metrics

Poisson runs P1 and P2 Lagrange elements on uniformly refined unit-square triangular meshes. The default mesh sizes are `n = 8, 16, 32, 64`. It writes `results/poisson_convergence.csv` with L2 errors, H1 seminorm errors, and adjacent-grid convergence rates.

Heat runs backward Euler to `T = 0.1` with paired mesh/time refinements. It writes `results/heat_convergence.csv` with final-time L2 errors.

Expected asymptotic behavior for smooth manufactured solutions:

- P1 Poisson: approximately second-order L2 error and first-order H1 seminorm error.
- P2 Poisson: approximately third-order L2 error and second-order H1 seminorm error.

Observed rates are not hard-coded. They are computed from generated CSV files.

<!-- GENERATED_RESULTS_START -->
Current observed numerical results: pending. Run `make reproduce` in an environment with DOLFINx available to generate CSV files, figures, and this summary block.
<!-- GENERATED_RESULTS_END -->

## Reproduction

Prerequisites:

- Python 3.10 or newer.
- GNU Make for the documented reproduction targets.
- Either an active DOLFINx/FEniCSx Python environment or Docker for the provided `dolfinx/dolfinx:stable` fallback.

Plain Python setup installs the package, tests, and plotting dependencies:

```bash
make setup
make test
```

If DOLFINx is already available in the active Python environment:

```bash
make poisson
make heat
make reproduce
```

If DOLFINx is not available locally, use the Docker fallback:

```bash
make docker-reproduce
```

The Docker path builds from `dolfinx/dolfinx:stable`, installs this package, and runs the same Makefile reproduction command.

If `make` is unavailable on Windows, install GNU Make through a tool such as MSYS2, Chocolatey, or WSL, then run the same targets above. The Python scripts under `scripts/` mirror those targets, but the Makefile remains the canonical reproduction interface.

## Cloud reproduction with GitHub Actions

This repository includes a GitHub Actions workflow at `.github/workflows/reproduce.yml` for running the full DOLFINx reproduction in the cloud with the official `dolfinx/dolfinx:stable` Docker image.

To run it manually:

1. Push the repository to GitHub.
2. Open the repository's **Actions** tab.
3. Select **Reproduce DOLFINx results**.
4. Click **Run workflow**.

The workflow installs the package, runs the tests, cleans generated outputs, runs `scripts/reproduce_all.py`, and then runs `scripts/check_outputs.py`. If the solver fails or any expected CSV/PNG output is missing, the workflow fails rather than fabricating or accepting incomplete results.

When the workflow succeeds, download the `fenics-generated-results` artifact from the workflow run page. It contains the generated `results/` CSV files and `figures/` PNG files.

## Expected Outputs

Poisson:

- `results/poisson_convergence.csv`
- `figures/poisson_solution.png`
- `figures/poisson_error.png`
- `figures/poisson_convergence.png`

Heat:

- `results/heat_convergence.csv`
- `figures/heat_initial_condition.png`
- `figures/heat_final_solution.png`
- `figures/heat_final_error.png`
- `figures/heat_error_trend.png`

## Limitations

This is a controlled manufactured-solution mini-project, not a production CFD or multiphysics solver. It uses simple unit-square geometry, homogeneous Dirichlet boundary conditions, serial plotting, and small refinement studies intended for verification and portfolio readability.
