"""Fast checks of the aircraft runner contract without invoking the solver."""

import tempfile
import unittest
from unittest.mock import Mock, patch

from ambiance import Atmosphere

from libraries.aerospace.geometry.aircraft_definition import Aircraft, Wing, WingXsec, Airfoil
from tests import test_vsp_run as runner_module


class AircraftRunnerTests(unittest.TestCase):
    def setUp(self):
        self.aircraft = Aircraft("Test", [Wing(
            "Wing", [WingXsec(3, 1.5, 0.5, -2, 0, 0, 0)],
            [Airfoil(0.12, 0.04, 0.4) for _ in range(2)], 0, 0, 0, 0, 0, 0,
        )])

    def test_bad_flight_conditions_rejected_before_loading_openvsp(self):
        with patch.object(runner_module, "_load_openvsp") as loader:
            for V, alpha in [(0, 5), (-1, 5), (30, float("nan"))]:
                with self.assertRaises(ValueError):
                    runner_module.vsp_run(self.aircraft, V, alpha)
            loader.assert_not_called()

    def test_discontinuous_sections_rejected_before_build(self):
        wing = self.aircraft.wing_list[0]
        wing.wing_sections.append(WingXsec(1, 0.6, 0.3, -3, 0, 0, 0))
        wing.airfoils.append(Airfoil(0.12, 0.04, 0.4))
        vsp = Mock()
        with self.assertRaisesRegex(ValueError, "share a chord"):
            runner_module._add_aircraft_wing(vsp, wing)
        vsp.AddGeom.assert_not_called()

    def test_velocity_altitude_and_references_feed_solver(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(runner_module, "_load_openvsp", return_value=Mock()), \
                patch.object(runner_module, "_add_aircraft_wing", return_value=("wing", 6, 6)), \
                patch.object(runner_module, "_run_sweep", return_value="result") as sweep, \
                patch.object(runner_module, "_read_results", return_value={
                    "Alpha": [5], "CL": [1], "CD": [0.04], "Cm": [0],
                }):
            result = runner_module.vsp_run(self.aircraft, 30, 5, h=1000, out_dir=folder)
        atmosphere = Atmosphere(1000)
        self.assertAlmostEqual(result["analysis"]["Mach"], 30 / atmosphere.speed_of_sound[0])
        self.assertAlmostEqual(result["analysis"]["ReCref"], 30 / atmosphere.kinematic_viscosity[0])
        self.assertEqual(sweep.call_args.args[2], [5])
        self.assertEqual(sweep.call_args.args[-2], 30)
        self.assertAlmostEqual(sweep.call_args.args[-1], atmosphere.density[0])
        self.assertEqual(result["reference"]["S_ref"], 6)


if __name__ == "__main__":
    unittest.main()
