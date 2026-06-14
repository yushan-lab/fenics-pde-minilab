from pathlib import Path

from scripts.check_outputs import EXPECTED_OUTPUTS, find_missing_outputs


def test_expected_outputs_match_reproduction_contract() -> None:
    assert EXPECTED_OUTPUTS == (
        Path("results/poisson_convergence.csv"),
        Path("results/poisson_error_summary.csv"),
        Path("results/heat_convergence.csv"),
        Path("figures/poisson_solution.png"),
        Path("figures/poisson_error.png"),
        Path("figures/poisson_convergence.png"),
        Path("figures/heat_initial_condition.png"),
        Path("figures/heat_final_solution.png"),
        Path("figures/heat_final_error.png"),
        Path("figures/heat_error_trend.png"),
    )


def test_find_missing_outputs_reports_absent_and_empty_files(tmp_path: Path) -> None:
    present = tmp_path / "results/poisson_convergence.csv"
    present.parent.mkdir(parents=True)
    present.write_text("degree,n,h,L2_error\n", encoding="utf-8")

    empty = tmp_path / "figures/poisson_solution.png"
    empty.parent.mkdir(parents=True)
    empty.touch()

    missing = find_missing_outputs(tmp_path)

    assert Path("results/poisson_convergence.csv") not in missing
    assert Path("figures/poisson_solution.png") in missing
    assert Path("results/heat_convergence.csv") in missing
