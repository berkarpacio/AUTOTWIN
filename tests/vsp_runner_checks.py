"""Run with `python tests/vsp_runner_checks.py`; OpenVSP is not required."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).with_name("test_vsp_run.py")
SPEC = importlib.util.spec_from_file_location("simplified_vsp_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerChecks(unittest.TestCase):
    def test_alpha_formats_and_order(self):
        for raw in ([2, 0, 1], "[2, 0, 1]", "2, 0, 1"):
            self.assertEqual(runner._read_alphas(raw), [2.0, 0.0, 1.0])
        self.assertEqual(runner._read_alphas(None), [0.0, 1.0, 2.0])
        self.assertEqual(runner._read_alphas("3"), [3.0])

    def test_invalid_sweep_angles(self):
        for raw in ([], True, [0, 1, 3], [0, 0], [float("nan")], [float("inf")]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner._read_alphas(raw)

    def test_results_filtering_order_duplicates_and_missing_moment(self):
        data = {
            "mesh": {"Nodes": [100]},
            "zero": {"Alpha": [0], "CLtot": [0], "CDtot": [0]},
            "alpha0": {"Alpha": [0, 0], "CLtot": [0.1, 0.3], "CDtot": [0.02], "CMytot": [-0.04]},
            "alpha2": {"AlphaDeg": [2], "CL": [0.5], "CD": [0.03]},
            "unrequested": {"Alpha": [9], "CL": [1], "CD": [0.1]},
        }
        vsp = Mock()
        vsp.GetStringResults.return_value = list(data)
        vsp.GetAllDataNames.side_effect = lambda key: list(data[key])
        vsp.GetDoubleResults.side_effect = lambda key, name: data[key][name]
        notes = []
        result = runner._read_results(vsp, "sweep", [2, 0, 1], notes)
        self.assertEqual(result, {"Alpha": [2, 0], "CL": [0.5, 0.3], "CD": [0.03, 0.02], "Cm": [None, -0.04]})
        self.assertTrue(any("Missing CMytot" in note for note in notes))
        self.assertTrue(any("alpha=1" in note for note in notes))

    def test_fuselage_resize_stops_when_api_does_nothing(self):
        vsp = Mock()
        vsp.GetNumXSec.return_value = 5
        with self.assertRaisesRegex(RuntimeError, "Could not resize"):
            runner._resize_fuselage(vsp, "fuselage", 6)

    def test_missing_parameter_tries_next_candidate(self):
        vsp = Mock()
        vsp.FindParm.side_effect = ["", "valid-id"]
        notes = []
        runner._set_parameter(vsp, ["wing"], ["first", "second"], ["group"], 0.3, notes)
        vsp.SetParmVal.assert_called_once_with("valid-id", 0.3)
        self.assertEqual(notes, [])

    def test_workflow_builds_all_components_but_analyzes_only_wings(self):
        vsp = Mock()
        for name, value in {
            "SET_NONE": -1, "SET_ALL": 0, "SET_SHOWN": 1, "SET_NOT_SHOWN": 2,
            "SPAN_WSECT_DRIVER": 0, "ROOTC_WSECT_DRIVER": 1, "TIPC_WSECT_DRIVER": 2,
            "XS_CIRCLE": 1, "XS_EDIT_CURVE": 2, "EDIT_XSEC_RECTANGLE": 3,
            "SS_CONTROL": 4,
        }.items():
            setattr(vsp, name, value)
        vsp.AddGeom.side_effect = ["fuselage", "boom", "wing", "tail", "prop"]
        vsp.GetNumXSec.return_value = 2
        vsp.GetNumSubSurf.return_value = 1
        vsp.GetXSecParm.return_value = "section-parameter"
        vsp.FindParm.return_value = "parameter"
        vsp.ListAnalysis.return_value = ["VSPAEROComputeGeometry", "VSPAEROSweep"]
        vsp.ExecAnalysis.side_effect = ["mesh-result", "sweep-result"]
        vsp.GetStringResults.return_value = ["alpha-result"]
        results = {"Alpha": [0], "CLtot": [0.3], "CDtot": [0.02], "CMytot": [-0.04]}
        vsp.GetAllDataNames.return_value = list(results)
        vsp.GetDoubleResults.side_effect = lambda key, name: results[name]
        inputs = {
            "cg_xyz": [0.25, 0, 0], "alpha_list_deg": [0],
            "wings": [
                {"S": 2, "AR": 8, "ctrl_start": 0.2, "ctrl_end": 0.8, "ctrl_hinge": 0.3},
                {"S": 0.5, "AR": 2},
            ],
            "fuselages": [
                {"length": 1, "XSec_0_w": 0.1, "XSec_0_h": 0.2, "XSec_1_w": 0.1, "XSec_1_h": 0.2},
                {"length": 1, "shape": "circle", "XSec_0_diameter": 0.1, "XSec_1_diameter": 0.1},
            ],
            "props": [{"radius": 0.2, "num_blades": 2}],
        }
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "_load_openvsp", return_value=vsp):
            output = runner.vsp_run(inputs, out_dir=folder, mach=0.1, reynolds_cref=500000)
            vsp.WriteVSPFile.assert_called_once_with(str(Path(folder).resolve() / "autotwin_aircraft.vsp3"), vsp.SET_ALL)
        self.assertEqual(output["reference"], {"Sref": 2.5, "Bref": 4.0, "Cref": 0.625, "cg": (0.25, 0, 0)})
        self.assertEqual(output["geom_ids"], {"fuselage": ["fuselage", "boom"], "wing": ["wing", "tail"], "prop": ["prop"]})
        for geom_id in ("wing", "tail"):
            vsp.SetSetFlag.assert_any_call(geom_id, vsp.SET_SHOWN, True)
        for geom_id in ("fuselage", "boom", "prop"):
            vsp.SetSetFlag.assert_any_call(geom_id, vsp.SET_SHOWN, False)
        self.assertEqual([call.args[0] for call in vsp.ExecAnalysis.call_args_list], ["VSPAEROComputeGeometry", "VSPAEROSweep"])
        vsp.SetDoubleAnalysisInput.assert_any_call("VSPAEROSweep", "ReCref", [500000.0], 0)
        vsp.SetDoubleAnalysisInput.assert_any_call("VSPAEROSweep", "MachStart", [0.1], 0)
        self.assertEqual(output["sweep_outputs"]["CL"], [0.3])
        self.assertEqual(output["analysis"], {"results_id": "sweep-result"})


if __name__ == "__main__":
    unittest.main()
