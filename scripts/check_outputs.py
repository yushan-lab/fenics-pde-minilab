from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_OUTPUTS = (
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


def find_missing_outputs(root: Path = Path(".")) -> list[Path]:
    """Return expected artifacts that are absent or empty."""
    missing: list[Path] = []
    for relative_path in EXPECTED_OUTPUTS:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative_path)
    return missing


def main() -> int:
    missing = find_missing_outputs(Path("."))
    if missing:
        print("Missing or empty generated output files:", file=sys.stderr)
        for path in missing:
            print(f"- {path.as_posix()}", file=sys.stderr)
        return 1

    print("All expected generated CSV and PNG files are present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
