from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from fenics_pde_minilab.poisson import run_poisson_convergence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Poisson convergence study.")
    parser.add_argument("--mesh-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()

    try:
        rows = run_poisson_convergence(mesh_sizes=args.mesh_sizes, degrees=args.degrees)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"Wrote {len(rows)} Poisson rows to {Path('results/poisson_convergence.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
