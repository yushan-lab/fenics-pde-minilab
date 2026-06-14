from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from fenics_pde_minilab.heat import run_heat_convergence
from fenics_pde_minilab.poisson import run_poisson_convergence
from fenics_pde_minilab.reporting import update_readme_results


def main() -> int:
    try:
        poisson_rows = run_poisson_convergence()
        heat_rows = run_heat_convergence()
        update_readme_results()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"Generated {len(poisson_rows)} Poisson rows and {len(heat_rows)} heat rows.")
    print("Updated README.md generated-results block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
