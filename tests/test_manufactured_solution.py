import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from fenics_pde_minilab.errors import (
    convergence_rates,
    heat_exact,
    poisson_exact,
    poisson_rhs,
)


def test_poisson_rhs_matches_manufactured_solution() -> None:
    x = np.array([0.25, 0.5, 0.75])
    y = np.array([0.2, 0.4, 0.6])

    expected = 2.0 * math.pi**2 * poisson_exact(x, y)

    np.testing.assert_allclose(poisson_rhs(x, y), expected)


def test_heat_exact_matches_initial_condition_and_decay() -> None:
    x = np.array([0.25, 0.5])
    y = np.array([0.5, 0.5])
    kappa = 0.4
    t = 0.125

    initial = poisson_exact(x, y)
    expected_decay = math.exp(-2.0 * math.pi**2 * kappa * t) * initial

    np.testing.assert_allclose(heat_exact(x, y, t=0.0, kappa=kappa), initial)
    np.testing.assert_allclose(heat_exact(x, y, t=t, kappa=kappa), expected_decay)


def test_convergence_rates_use_adjacent_refinements() -> None:
    mesh_sizes = np.array([1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0])
    errors = mesh_sizes**2

    rates = convergence_rates(mesh_sizes, errors)

    assert math.isnan(rates[0])
    np.testing.assert_allclose(rates[1:], [2.0, 2.0])


def test_script_help_runs_from_fresh_checkout_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script in ("scripts/run_poisson.py", "scripts/run_heat.py"):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
