"""Optimization contract checks without OpenVSP."""

import unittest
from unittest.mock import Mock

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from libraries.aerospace.optimization.VSPAERO_optimization import (
    VSPAeroOptimProblem, wing_aero_shape_optimization,
)


def fake_runner(aircraft, V, alpha, *, h=0):
    return {"sweep_outputs": {"CL": [alpha / 5], "CD": [0.02 + alpha**2 / 1000]}}


class WingOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.runner = Mock(side_effect=fake_runner)
        self.problem = VSPAeroOptimProblem(30, 100, 6, 6, self.runner)
        self.x = np.array([5, 1.5, -1, 1.2, -2, 0.8, -3])

    def test_geometry_and_station_twists(self):
        result = self.problem.evaluate(self.x, return_as_dictionary=True)
        self.assertTrue(np.all(result["G"] <= 0))
        wing = self.runner.call_args.args[0].wing_list[0]
        self.assertAlmostEqual(2 * sum(s.span for s in wing.wing_sections), 6)
        self.assertAlmostEqual(sum(s.span * (s.root_chord + s.tip_chord)
                                   for s in wing.wing_sections), 6)
        self.assertEqual([s.twist_deg for s in wing.wing_sections], [-1, -2, -3])
        self.assertAlmostEqual(wing.wing_sections[-1].tip_chord, 0.5)

    def test_each_reverse_taper_is_rejected_without_solver(self):
        for i, chords in enumerate([(1, 1.2, 1), (1.5, 0.8, 1), (1.5, 1, 0.7)]):
            with self.subTest(section=i):
                x = self.x.copy()
                x[[1, 3, 5]] = chords
                result = self.problem.evaluate(x, return_as_dictionary=True)
                self.assertGreater(result["G"][i], 0)
        self.runner.assert_not_called()

    def test_equal_chords_and_negative_derived_tip_rejected(self):
        for chords in [(1, 1, 1), (2, 2, 2)]:
            x = self.x.copy()
            x[[1, 3, 5]] = chords
            result = self.problem.evaluate(x, return_as_dictionary=True)
            self.assertTrue(np.any(result["G"] > 0))
        self.runner.assert_not_called()

    def test_nonuniform_stations_preserve_area(self):
        problem = VSPAeroOptimProblem(30, 100, 6, 6, self.runner,
                                     station_fractions=(0, 0.2, 0.6, 1))
        x = self.x.copy()
        x[5] = 0.9
        result = problem.evaluate(x, return_as_dictionary=True)
        self.assertTrue(np.all(result["G"] <= 0))
        sections = self.runner.call_args.args[0].wing_list[0].wing_sections
        self.assertAlmostEqual(sum(s.span * (s.root_chord + s.tip_chord)
                                   for s in sections), 6)

    def test_lift_and_solver_failures(self):
        x = self.x.copy()
        x[0] = 4
        result = self.problem.evaluate(x, return_as_dictionary=True)
        self.assertGreater(result["G"][-1], 0)
        for output in [None, {"sweep_outputs": {"CL": [1, 2], "CD": [0.1]}},
                       {"sweep_outputs": {"CL": [1], "CD": [np.nan]}}]:
            self.runner.side_effect = None
            self.runner.return_value = output
            result = self.problem.evaluate(self.x, return_as_dictionary=True)
            self.assertTrue(np.isinf(result["F"][0]))
            self.assertGreater(result["G"][-1], 0)
        self.assertEqual(self.problem.failed_evaluations, 3)

    def test_real_pymoo_loop_retains_feasible_candidate(self):
        sampling = np.tile(self.x, (4, 1))
        sampling[:, 0] = [5, 4, 3, 2]
        result = minimize(self.problem, GA(pop_size=4, sampling=sampling),
                          termination=("n_gen", 3), seed=1, verbose=False)
        self.assertIsNotNone(result.X)
        self.assertTrue(np.all(result.G <= 0))
        self.assertAlmostEqual(result.X[0], 5, places=2)

    def test_public_wrapper_and_failed_run(self):
        runner = Mock(side_effect=fake_runner)
        result = wing_aero_shape_optimization(
            30, 100, 6, 6, runner=runner, CL_tolerance=2,
            pop_size=40, n_gen=5, verbose=False,
        )
        self.assertIsInstance(result, np.ndarray)
        chords = result[[1, 3, 5, 7]]
        self.assertTrue(np.all(np.diff(chords) <= -0.001))
        self.assertAlmostEqual(sum(chords[:-1] + chords[1:]), 6)
        # The last solver call must rebuild the selected design for viewing.
        self.assertEqual(runner.call_args.kwargs["alpha"], result[0])
        sections = runner.call_args.args[0].wing_list[0].wing_sections
        np.testing.assert_allclose([s.root_chord for s in sections], chords[:-1])
        np.testing.assert_allclose([s.tip_chord for s in sections], chords[1:])
        np.testing.assert_allclose([s.twist_deg for s in sections], result[[2, 4, 6]])
        failure = wing_aero_shape_optimization(
            30, 100, 6, 6, runner=Mock(side_effect=RuntimeError("solver failed")),
            pop_size=40, n_gen=2, verbose=False,
        )
        self.assertEqual(failure["status"], "no_solution_found")
        self.assertGreater(failure["failed_evaluations"], 0)
        self.assertEqual(failure["last_error"], "solver failed")


if __name__ == "__main__":
    unittest.main()
