"""Workflow checks without loading OpenVSP: python -m unittest tests.test_aero_shape_optim_vsp."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from reference_scripts.aero_shape_optim_vsp import (
    BASELINE, Settings, WingProblem, build_inputs, read_coefficients,
    ConvergenceMonitor, save_progress,
)


def fake_solver(inputs, **kwargs):
    wing = inputs["wings"][0]
    alpha = inputs["alpha_list_deg"][0]
    cl = alpha / 10
    cd = 0.01 + cl ** 2 / (np.pi * wing["AR"]) + 0.01 * (wing["taper"] - 0.4) ** 2
    return {"sweep_outputs": {"Alpha": [alpha], "CL": [cl], "CD": [cd]},
            "vsp3_path": str(Path(kwargs["out_dir"]) / "fake.vsp3"), "notes": []}


class ShapeOptimizationChecks(unittest.TestCase):
    def test_fixed_area_span_and_monotonic_chord(self):
        settings = Settings()
        for taper in (0.15, 0.5, 1):
            wing = build_inputs([taper, 5, 10], settings)["wings"][0]
            span = np.sqrt(wing["S"] * wing["AR"])
            root = 2 * wing["S"] / (span * (1 + taper))
            tip = root * taper
            self.assertAlmostEqual(span, settings.span)
            self.assertAlmostEqual(span * (root + tip) / 2, settings.area)
            self.assertTrue(0 < tip <= root)

    def test_invalid_results_rejected(self):
        for sweep in ({"Alpha": [], "CL": [], "CD": []},
                      {"Alpha": [9], "CL": [1], "CD": [0.03]},
                      {"Alpha": [10], "CL": [float("nan")], "CD": [0.03]},
                      {"Alpha": [10], "CL": [1], "CD": [-0.1]}):
            with self.subTest(sweep=sweep), self.assertRaises(ValueError):
                read_coefficients({"sweep_outputs": sweep}, 10)

    def test_cache_constraints_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = Mock(side_effect=fake_solver)
            problem = WingProblem(Settings(), folder, runner)
            good = problem.evaluate_design(BASELINE)
            self.assertTrue(good["feasible"])
            self.assertEqual(good, problem.evaluate_design(BASELINE.copy()))
            runner.assert_called_once()
            low_lift = problem.evaluate_design([0.5, 0, 8])
            self.assertGreater(low_lift["constraint"], 0)
            rows = Path(folder, "history.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0])["status"], "ok")

    def test_failures_penalized_and_preflight_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            problem = WingProblem(Settings(), folder, Mock(side_effect=RuntimeError("solver failed")))
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                problem.evaluate_design(BASELINE, strict=True)
        with tempfile.TemporaryDirectory() as folder:
            problem = WingProblem(Settings(), folder, Mock(side_effect=RuntimeError("solver failed")))
            result = problem.evaluate_design(BASELINE)
            self.assertEqual(result["status"], "failed")
            self.assertGreater(result["constraint"], 0)
            self.assertGreater(result["objective"], 100)

    def test_pymoo_loop_and_feasible_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            problem = WingProblem(Settings(), folder, fake_solver)
            sampling = np.array([BASELINE, [0.3, 0, 10], [0.8, 5, 8], [0.7, 0, 12]])
            result = minimize(problem, GA(pop_size=4, sampling=sampling),
                              termination=("n_gen", 2), seed=1, callback=ConvergenceMonitor(2))
            self.assertIsNotNone(result.X)
            self.assertLessEqual(float(result.G[0]), 0)
            self.assertGreater(len(problem.records), 4)

    def test_report_ignores_lower_drag_infeasible_design(self):
        with tempfile.TemporaryDirectory() as folder:
            problem = WingProblem(Settings(), folder, fake_solver)
            feasible = problem.evaluate_design(BASELINE)
            problem.evaluate_design([0.5, 0, 1])
            best = save_progress(folder, list(problem.records.values()), "interrupted")
            self.assertEqual(best, feasible)
            self.assertTrue(Path(folder, "convergence.png").is_file())
            self.assertTrue(Path(folder, "convergence.csv").is_file())
            summary = json.loads(Path(folder, "summary.json").read_text())
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["evaluations"], 2)

    def test_stagnation_stops_only_after_feasibility(self):
        with tempfile.TemporaryDirectory() as folder:
            problem = WingProblem(Settings(), folder, fake_solver)
            monitor = ConvergenceMonitor(30, patience=2)
            algorithm = Mock(problem=problem, n_gen=1)
            algorithm.pop.get.return_value = np.array([[1]])
            algorithm.pop.__len__ = Mock(return_value=1)
            problem.evaluate_design([0.5, 0, 1])
            for _ in range(3):
                monitor.notify(algorithm)
            algorithm.termination.terminate.assert_not_called()
            problem.evaluate_design(BASELINE)
            for _ in range(3):
                monitor.notify(algorithm)
            algorithm.termination.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
