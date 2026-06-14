import ast
import csv
import re
from pathlib import Path

from fenics_pde_minilab.heat import HeatResult, _rows as heat_rows
from fenics_pde_minilab.poisson import PoissonResult, _rows_with_rates
from fenics_pde_minilab.poisson import write_poisson_error_summary
from fenics_pde_minilab.reporting import build_results_summary


ROOT = Path(__file__).resolve().parents[1]


def test_poisson_rates_are_computed_from_result_errors() -> None:
    rows = _rows_with_rates(
        [
            PoissonResult(degree=1, n=8, h=1.0 / 8.0, l2_error=(1.0 / 8.0) ** 2, h1_seminorm_error=1.0 / 8.0),
            PoissonResult(degree=1, n=16, h=1.0 / 16.0, l2_error=(1.0 / 16.0) ** 2, h1_seminorm_error=1.0 / 16.0),
            PoissonResult(degree=1, n=32, h=1.0 / 32.0, l2_error=(1.0 / 32.0) ** 2, h1_seminorm_error=1.0 / 32.0),
        ]
    )

    assert rows[1]["L2_rate"] == 2.0
    assert rows[2]["L2_rate"] == 2.0
    assert rows[1]["H1_rate"] == 1.0
    assert rows[2]["H1_rate"] == 1.0


def test_poisson_error_summary_schema_contains_sampled_linf(tmp_path: Path) -> None:
    result = PoissonResult(
        degree=2,
        n=64,
        h=1.0 / 64.0,
        l2_error=1.0e-6,
        h1_seminorm_error=1.0e-4,
    )
    path = tmp_path / "poisson_error_summary.csv"

    write_poisson_error_summary(result, 2.0e-6, path)

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows == [
        {
            "degree": "2",
            "n": "64",
            "L2_error": "1e-06",
            "H1_seminorm_error": "0.0001",
            "sampled_Linf_error": "2e-06",
        }
    ]


def test_heat_rows_include_adjacent_l2_rates() -> None:
    rows = heat_rows(
        [
            HeatResult(n=8, steps=20, h=1 / 8, dt=0.005, theta=0.5, final_time=0.1, kappa=1.0, final_l2_error=(1 / 8) ** 2),
            HeatResult(n=16, steps=40, h=1 / 16, dt=0.0025, theta=0.5, final_time=0.1, kappa=1.0, final_l2_error=(1 / 16) ** 2),
            HeatResult(n=32, steps=80, h=1 / 32, dt=0.00125, theta=0.5, final_time=0.1, kappa=1.0, final_l2_error=(1 / 32) ** 2),
        ]
    )

    assert rows[0]["L2_rate"] != rows[0]["L2_rate"]
    assert rows[1]["L2_rate"] == 2.0
    assert rows[2]["L2_rate"] == 2.0


def test_readme_result_summary_is_driven_by_csv_values(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    with (results_dir / "poisson_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["degree", "n", "h", "L2_error", "H1_seminorm_error", "L2_rate", "H1_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "degree": "1",
                "n": "8",
                "h": "0.125",
                "L2_error": "0.1",
                "H1_seminorm_error": "0.2",
                "L2_rate": "nan",
                "H1_rate": "nan",
            }
        )
        writer.writerow(
            {
                "degree": "1",
                "n": "16",
                "h": "0.0625",
                "L2_error": "0.01",
                "H1_seminorm_error": "0.05",
                "L2_rate": "7.125",
                "H1_rate": "3.5",
            }
        )

    with (results_dir / "poisson_error_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["degree", "n", "L2_error", "H1_seminorm_error", "sampled_Linf_error"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "degree": "1",
                "n": "16",
                "L2_error": "0.01",
                "H1_seminorm_error": "0.05",
                "sampled_Linf_error": "0.02",
            }
        )

    with (results_dir / "heat_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["n", "steps", "h", "dt", "theta", "final_time", "kappa", "final_L2_error", "L2_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n": "8",
                "steps": "20",
                "h": "0.125",
                "dt": "0.005",
                "theta": "0.5",
                "final_time": "0.1",
                "kappa": "1.0",
                "final_L2_error": "0.2",
                "L2_rate": "nan",
            }
        )
        writer.writerow(
            {
                "n": "16",
                "steps": "40",
                "h": "0.0625",
                "dt": "0.0025",
                "theta": "0.5",
                "final_time": "0.1",
                "kappa": "1.0",
                "final_L2_error": "0.05",
                "L2_rate": "2.0",
            }
        )

    summary = build_results_summary(results_dir)

    assert "| Element | Finest n | L2 error | H1 seminorm error | Final L2 rate | Final H1 rate |" in summary
    assert "7.125" in summary
    assert "3.5" in summary
    assert "sampled Linf" in summary
    assert "0.2 to 0.05" in summary
    assert "| n | Steps | Final-time L2 error | L2 rate |" in summary


def test_readme_documents_generated_results_without_pending_language() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "results pending" not in readme.lower()
    assert "Current observed numerical results: pending" not in readme
    for figure_path in (
        "figures/poisson_convergence.png",
        "figures/poisson_solution.png",
        "figures/poisson_error.png",
        "figures/heat_error_trend.png",
        "figures/heat_initial_condition.png",
        "figures/heat_final_solution.png",
        "figures/heat_final_error.png",
    ):
        assert figure_path in readme


def test_solver_sources_use_explicit_quadrature_for_exact_solution_integrals() -> None:
    assert "quadrature_degree" in (ROOT / "src/fenics_pde_minilab/poisson.py").read_text(encoding="utf-8")
    assert "quadrature_degree" in (ROOT / "src/fenics_pde_minilab/heat.py").read_text(encoding="utf-8")


def test_poisson_error_plot_uses_dense_exact_sampling_and_writes_summary() -> None:
    poisson_source = (ROOT / "src/fenics_pde_minilab/poisson.py").read_text(encoding="utf-8")
    plotting_source = (ROOT / "src/fenics_pde_minilab/plotting.py").read_text(encoding="utf-8")

    assert "_p1_solution_and_error" not in poisson_source
    assert "dense_grid_scalar_fields_from_dolfinx_function" in poisson_source
    assert "poisson_error_summary.csv" in poisson_source
    assert "sampled_Linf_error" in poisson_source
    assert "compute_collisions_points" in plotting_source
    assert "function.eval" in plotting_source
    assert "exact_function(points_xy[:, 0], points_xy[:, 1])" in plotting_source


def test_heat_solver_uses_crank_nicolson_refinement_and_rates() -> None:
    heat_source = (ROOT / "src/fenics_pde_minilab/heat.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    method_notes = (ROOT / "docs/method_notes.md").read_text(encoding="utf-8")

    assert "theta: float = 0.5" in heat_source
    assert "theta * dt * kappa" in heat_source
    assert "(1.0 - theta) * dt * kappa" in heat_source
    assert "L2_rate" in heat_source
    assert "((8, 20), (16, 40), (32, 80), (64, 160))" in heat_source
    assert "Crank-Nicolson" in readme
    assert "Crank-Nicolson" in method_notes


def test_all_linear_problems_use_current_dolfinx_options_prefix_api() -> None:
    solver_paths = [
        ROOT / "src/fenics_pde_minilab/poisson.py",
        ROOT / "src/fenics_pde_minilab/heat.py",
    ]

    calls: list[tuple[Path, ast.Call]] = []
    for path in solver_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LinearProblem":
                calls.append((path, node))

    assert calls
    for path, call in calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "petsc_options" in keyword_names, path
        assert "petsc_options_prefix" in keyword_names, path

    poisson_source = (ROOT / "src/fenics_pde_minilab/poisson.py").read_text(encoding="utf-8")
    heat_source = (ROOT / "src/fenics_pde_minilab/heat.py").read_text(encoding="utf-8")
    assert 'f"poisson_p{degree}_n{n}_"' in poisson_source
    assert 'f"heat_p{degree}_n{n}_dt{steps}_"' in heat_source


def test_readme_figure_paths_are_generated_by_repository_scripts() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    figure_paths = set(re.findall(r"figures/[A-Za-z0-9_]+\.png", readme))
    script_sources = "\n".join(
        [
            (ROOT / "src/fenics_pde_minilab/poisson.py").read_text(encoding="utf-8"),
            (ROOT / "src/fenics_pde_minilab/heat.py").read_text(encoding="utf-8"),
        ]
    )

    assert figure_paths
    for figure_path in figure_paths:
        assert Path(figure_path).name in script_sources


def test_cv_bullets_do_not_claim_completed_numerical_reproduction() -> None:
    text = (ROOT / "docs/cv_bullets.md").read_text(encoding="utf-8").lower()

    forbidden_phrases = [
        "generated convergence csv",
        "generated matplotlib figures",
        "observed convergence",
        "achieved convergence",
        "completed numerical reproduction",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in text
