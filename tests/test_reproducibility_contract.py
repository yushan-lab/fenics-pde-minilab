import ast
import csv
import re
from pathlib import Path

from fenics_pde_minilab.poisson import PoissonResult, _rows_with_rates
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

    with (results_dir / "heat_convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["n", "steps", "h", "dt", "final_time", "kappa", "final_L2_error"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n": "8",
                "steps": "20",
                "h": "0.125",
                "dt": "0.005",
                "final_time": "0.1",
                "kappa": "1.0",
                "final_L2_error": "0.2",
            }
        )
        writer.writerow(
            {
                "n": "16",
                "steps": "40",
                "h": "0.0625",
                "dt": "0.0025",
                "final_time": "0.1",
                "kappa": "1.0",
                "final_L2_error": "0.05",
            }
        )

    summary = build_results_summary(results_dir)

    assert "7.125" in summary
    assert "3.5" in summary
    assert "0.2 to 0.05" in summary


def test_solver_sources_use_explicit_quadrature_for_exact_solution_integrals() -> None:
    assert "quadrature_degree" in (ROOT / "src/fenics_pde_minilab/poisson.py").read_text(encoding="utf-8")
    assert "quadrature_degree" in (ROOT / "src/fenics_pde_minilab/heat.py").read_text(encoding="utf-8")


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
