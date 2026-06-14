from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from fenics_pde_minilab.heat import run_heat_convergence


def main() -> int:
    parser = argparse.ArgumentParser(description="Run heat-equation convergence study.")
    parser.add_argument("--mesh-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--steps", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()

    if len(args.mesh_sizes) != len(args.steps):
        print("--mesh-sizes and --steps must have the same length", file=sys.stderr)
        return 2

    try:
        rows = run_heat_convergence(
            cases=tuple(zip(args.mesh_sizes, args.steps, strict=True)),
            final_time=args.final_time,
            kappa=args.kappa,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"Wrote {len(rows)} heat rows to {Path('results/heat_convergence.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
