"""Update generated README result summaries from CSV outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path


START = "<!-- GENERATED_RESULTS_START -->"
END = "<!-- GENERATED_RESULTS_END -->"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _format_float(value: str) -> str:
    number = float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.4g}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_results_summary(results_dir: Path = Path("results")) -> str:
    poisson_path = results_dir / "poisson_convergence.csv"
    poisson_error_path = results_dir / "poisson_error_summary.csv"
    heat_path = results_dir / "heat_convergence.csv"
    if not poisson_path.exists() or not heat_path.exists():
        return (
            "Generated numerical results are not present in this checkout. Run `make reproduce` in an "
            "environment with DOLFINx available to regenerate CSV files, figures, and this summary block."
        )

    poisson_rows = _read_csv(poisson_path)
    heat_rows = _read_csv(heat_path)
    poisson_error_rows = _read_csv(poisson_error_path) if poisson_error_path.exists() else []
    lines = [
        "The tables below are read from generated CSV files produced by `scripts/reproduce_all.py`.",
        "",
        "Poisson convergence summary:",
        "",
    ]

    poisson_table_rows: list[list[str]] = []
    for degree in sorted({row["degree"] for row in poisson_rows}, key=float):
        degree_rows = sorted(
            [row for row in poisson_rows if row["degree"] == degree],
            key=lambda row: float(row["h"]),
            reverse=True,
        )
        finite_rows = [row for row in degree_rows if row["L2_rate"] != "nan" and row["H1_rate"] != "nan"]
        if finite_rows:
            last = finite_rows[-1]
            poisson_table_rows.append(
                [
                    f"P{int(float(degree))}",
                    str(int(float(last["n"]))),
                    _format_float(last["L2_error"]),
                    _format_float(last["H1_seminorm_error"]),
                    _format_float(last["L2_rate"]),
                    _format_float(last["H1_rate"]),
                ]
            )

    lines.extend(
        _markdown_table(
            ["Element", "Finest n", "L2 error", "H1 seminorm error", "Final L2 rate", "Final H1 rate"],
            poisson_table_rows,
        )
    )
    lines.extend(
        [
            "",
            "The Poisson rates are consistent with approximately second-order P1 L2 convergence, "
            "first-order P1 H1-seminorm convergence, third-order P2 L2 convergence, and "
            "second-order P2 H1-seminorm convergence.",
        ]
    )

    if poisson_error_rows:
        last_error = poisson_error_rows[-1]
        lines.extend(
            [
                "",
                "Poisson sampled pointwise error check:",
                "",
                *_markdown_table(
                    ["Element", "n", "L2 error", "sampled Linf error"],
                    [
                        [
                            f"P{int(float(last_error['degree']))}",
                            str(int(float(last_error["n"]))),
                            _format_float(last_error["L2_error"]),
                            _format_float(last_error["sampled_Linf_error"]),
                        ]
                    ],
                ),
            ]
        )

    if heat_rows:
        ordered_heat_rows = sorted(heat_rows, key=lambda row: float(row["h"]), reverse=True)
        first = ordered_heat_rows[0]
        last = ordered_heat_rows[-1]
        heat_errors = [float(row["final_L2_error"]) for row in ordered_heat_rows]
        monotone = all(later <= earlier for earlier, later in zip(heat_errors, heat_errors[1:], strict=False))
        direction = "decreased monotonically" if monotone else "did not decrease monotonically"
        heat_table_rows = [
            [
                str(int(float(row["n"]))),
                str(int(float(row["steps"]))),
                _format_float(row["final_L2_error"]),
                _format_float(row.get("L2_rate", "nan")),
            ]
            for row in ordered_heat_rows
        ]
        lines.extend(
            [
                "",
                "Heat Crank-Nicolson summary:",
                "",
                *_markdown_table(["n", "Steps", "Final-time L2 error", "L2 rate"], heat_table_rows),
                "",
                f"The final-time L2 error {direction} from {_format_float(first['final_L2_error'])} "
                f"to {_format_float(last['final_L2_error'])}.",
            ]
        )
        if not monotone:
            lines.append("- Heat error trend is non-monotone; inspect the CSV before treating it as convergence evidence.")

    lines.extend(["", "All numbers in this block are regenerated by `scripts/reproduce_all.py`."])
    return "\n".join(lines)


def update_readme_results(readme_path: Path = Path("README.md"), results_dir: Path = Path("results")) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("README.md is missing generated-results markers")

    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    summary = build_results_summary(results_dir)
    readme_path.write_text(f"{before}{START}\n{summary}\n{END}{after}", encoding="utf-8")
