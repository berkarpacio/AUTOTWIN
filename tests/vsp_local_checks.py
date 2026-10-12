"""Offline compatibility checks: python tests/vsp_local_checks.py."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "vsp_local", Path(__file__).resolve().parents[1] / "libraries/aerospace/aerodynamics/vsp_run_local.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class CompatibilityChecks(unittest.TestCase):
    def test_binary_detects_python_and_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "_vsp.pyd"
            data = bytearray(128)
            data[:2] = b"MZ"
            struct.pack_into("<I", data, 60, 64)
            data[64:68] = b"PE\0\0"
            struct.pack_into("<H", data, 68, 0x8664)
            path.write_bytes(data + b"\0python39.dll\0")
            self.assertEqual(runner._binary_info(path), (0x8664, ["python39.dll"]))

    def test_mismatch_rejected_before_native_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "python/openvsp/openvsp"
            package.mkdir(parents=True)
            for name in ("__init__.py", "vsp.py"):
                (package / name).touch()
            (package / "_vsp.pyd").write_bytes(b"\0python399.dll\0")
            with patch.object(runner.importlib, "import_module") as load:
                with self.assertRaisesRegex(RuntimeError, "binary mismatch"):
                    runner._load_openvsp(root)
                load.assert_not_called()

    def test_missing_configuration_is_actionable(self):
        with patch.dict(runner.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENVSP_ROOT"):
                runner._load_openvsp()

    def test_architecture_mismatch(self):
        with patch.object(runner, "_binary_info", side_effect=[(0x14c, []), (0x8664, [])]):
            with patch.object(runner.os, "name", "nt"):
                with self.assertRaisesRegex(RuntimeError, "Architecture mismatch"):
                    runner._check_binary(Path("unused.pyd"))

    def test_old_and_new_analysis_sets(self):
        for modern in (False, True):
            with self.subTest(modern=modern):
                api = Mock()
                api.SET_NONE, api.SET_SHOWN, api.VORTEX_LATTICE = -1, 1, 0
                api.ListAnalysis.return_value = ["VSPAEROComputeGeometry", "VSPAEROSweep"]
                inputs = ["GeomSet", "AnalysisMethod", "RefFlag", "WingID", "Symmetry", "Sref",
                          "bref", "cref", "Xcg", "Ycg", "Zcg", "ReCref", "MachStart", "MachEnd",
                          "AlphaStart", "AlphaEnd", "MachNpts", "AlphaNpts"]
                if modern:
                    inputs.append("ThinGeomSet")
                api.GetAnalysisInputNames.return_value = inputs
                api.ExecAnalysis.side_effect = ["mesh", "sweep"]
                result = runner._run_sweep(api, "wing", [0, 1], .1, 100000,
                                          {"cg": (0, 0, 0), "Sref": 1, "Bref": 2, "Cref": .5})
                self.assertEqual(result, "sweep")
                for name in api.ListAnalysis.return_value:
                    api.SetIntAnalysisInput.assert_any_call(name, "GeomSet", [-1 if modern else 1], 0)
                    if not modern:
                        api.SetIntAnalysisInput.assert_any_call(name, "AnalysisMethod", [0], 0)
                for call in api.SetDoubleAnalysisInput.call_args_list:
                    self.assertIn(call.args[1], inputs)


if __name__ == "__main__":
    unittest.main()
