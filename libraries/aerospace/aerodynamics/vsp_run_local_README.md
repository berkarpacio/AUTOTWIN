# Run the configurator on an air-gapped Windows computer

## Likely cause

OpenVSP's Python API contains a compiled `_vsp.pyd`. Its Python major/minor
version and architecture must match the interpreter running the script.
Installing NumPy/pandas or changing the OpenVSP path does not fix that mismatch.
The official archive lists OpenVSP 3.35.3 Windows builds for Python 3.6 and 3.9,
not 3.13. Without the original traceback and exact package, this remains the
leading diagnosis rather than a confirmed diagnosis of that computer.

Other possible causes are an incomplete OpenVSP copy, missing Microsoft Visual
C++ runtime, a different Python selected by VS Code, or mixed OpenVSP DLLs.
The current repository configurator also imports `tests.test_vsp_run`; this
import is unsuitable when only two files are copied into a flat folder.

Sources:

- [OpenVSP Python requirements](https://openvsp.org/pyapi_docs/latest/)
- [Official older OpenVSP packages](https://openvsp.org/download_old.php)
- [Windows Python DLL search behavior](https://docs.python.org/3/library/os.html#os.add_dll_directory)
- [Microsoft Visual C++ runtime installers](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist/)

## Prepare the software

Choose one compatible pair before transferring files:

1. Keep Python 3.13 and transfer a complete Windows OpenVSP ZIP explicitly built
   for Python 3.13 (and the same architecture). This runner was tested with
   OpenVSP 3.51.3 / standard 64-bit CPython 3.13.12.
2. Keep OpenVSP 3.35 and use the Python specified by its package. For the
   3.35.3 **Python3.9** win64 ZIP, install standard 64-bit CPython 3.9 alongside
   3.13 and select that interpreter. This runner supports Python 3.8+; if the
   ZIP says Python3.6, transfer the Python3.9 OpenVSP build and matching Python
   instead. Do not just rename or copy a Python DLL.

Transfer/extract the entire OpenVSP distribution, including `python`, executables,
and DLLs. Preserve its directory structure. Transfer the Microsoft Visual C++
redistributable required by that build if missing, using an official installer
compatible with the target Windows version and architecture. VS Code and its
extensions do not supply the OpenVSP runtime.

The runner itself uses only Python's standard library and the core OpenVSP API.
It disables optional OpenVSP helper imports where supported. If a particular
OpenVSP build still reports a missing Python dependency, obtain its compatible
wheel and dependencies on the connected computer, then install them with the
selected interpreter using `python -m pip install --no-index --find-links <wheel-folder> <package>`.
The runner never downloads anything or installs software.

## The two files

Copy these files into your offline folder:

```text
vsp_test_local/
    test_configurator.py
    vsp_run_local.py
```

In the **offline copy** of `test_configurator.py`, replace:

```python
from tests.test_vsp_run import vsp_run
```

(or the existing `from libraries...vsp_run import vsp_run`) with:

```python
from vsp_run_local import vsp_run
```

Keep your geometry calculations and `input_dict`. Immediately before its last
`output = vsp_run(...)` line, add the actual extracted OpenVSP root:

```python
input_dict["openvsp_root"] = r"C:\Tools\OpenVSP-3.35.3-win64-Python3.9"
output = vsp_run(input_dict=input_dict)
print(output["sweep_outputs"])
print(output["notes"])
```

That path is an example: use the directory containing your `python` folder and
OpenVSP executables. A ZIP may extract to a directory with a different name.
The repository's `ROOT_DIR` / `sys.path` setup can be removed from the offline
configurator; this runner has no imports from the repository.

Alternatively configure `OPENVSP_ROOT` in the environment or edit
`DEFAULT_OPENVSP_ROOT` in `vsp_run_local.py`. Priority is input dictionary,
environment, then file default. Restart Python after changing installations.

## Check, then run

In PowerShell inside `vsp_test_local`, run the check with the chosen interpreter.
For example, if using a Python 3.9 OpenVSP build and the Windows Python launcher:

```powershell
py -3.9 vsp_run_local.py --openvsp-root 'C:\Tools\OpenVSP-3.35.3-win64-Python3.9'
py -3.9 test_configurator.py
```

If `py` is unavailable, use the full Python executable path:

```powershell
& 'C:\Path\To\Python39\python.exe' vsp_run_local.py --openvsp-root 'C:\Tools\OpenVSP-3.35.3-win64-Python3.9'
& 'C:\Path\To\Python39\python.exe' test_configurator.py
```

For Python 3.13 builds substitute the corresponding interpreter. Select that
same interpreter in VS Code. The diagnostic prints the executable actually in
use, OpenVSP module path, and available analyses. It checks Python DLL references,
PE architecture, import paths, and solver discovery. An unrecognized binary
without an embedded Python DLL name is still subject to the native loader's
compatibility checks. Successful import does not prove the solver can run.

To save a check report for later inspection:

```powershell
py -3.9 vsp_run_local.py --openvsp-root 'C:\Tools\OpenVSP-3.35.3-win64-Python3.9' *> openvsp_check.txt
```

## Analysis behavior and validation

The geometry and return format are based on the standalone runner currently
called by this repository's configurator. The saved model contains fuselages,
wings/tails, and propellers. Aerodynamics uses **wings/tails only**, excluding
fuselages and propellers, just like that runner. This is not a whole-aircraft
drag prediction. Angles must be distinct and evenly spaced.

Older API inputs (`AnalysisMethod`/`GeomSet` with VLM) and newer inputs
(`ThinGeomSet`) are selected by inspecting the installed API. Missing essential
inputs, empty analysis IDs, and incomplete sweep results raise errors instead
of returning apparent success. The runner does not guarantee compatibility
with every older geometry API or eliminate the need for compatible binaries.

Validation performed on the connected computer:

- Five automated checks passed: binary metadata, Python mismatch before import,
  architecture mismatch, missing configuration, and old/new analysis selection.
- Real OpenVSP 3.51.3 / CPython 3.13.12 import and solver discovery passed.
- Real configurator geometry and sweep at 0, 1, 2 degrees completed. Returned
  CL = [0.33163, 0.41430, 0.49687] and CD = [0.01323, 0.01597, 0.01930].
- OpenVSP 3.35 and the air-gapped machine have **not** been run or validated.

Run the automated checks in the repository with `python tests/vsp_local_checks.py`.
