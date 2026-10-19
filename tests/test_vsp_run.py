"""Aircraft-object VSPAERO runner; importing this module does not run OpenVSP."""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import sys


# Edit this path when moving or upgrading your OpenVSP installation.
DEFAULT_OPENVSP_ROOT = Path(r"C:\Users\Berk\OneDrive\Desktop\OpenVSP-3.51.3-win64")

# Keep these handles alive: closing one removes its Windows DLL search path.
_DLL_HANDLES = {}


def _load_openvsp(root=None):
    """Configure an optional installation, then load its Python bindings."""
    if root:
        root = Path(root).expanduser().resolve()
        python_dir = root / "python"
        if not python_dir.is_dir():
            raise FileNotFoundError(f"OpenVSP Python folder not found: {python_dir}")
        exe_dir = next(
            (folder for folder in (root, root / "bin")
             if any((folder / exe).is_file() for exe in ("vsp.exe", "vspaero.exe"))),
            None,
        )
        if exe_dir is None:
            raise FileNotFoundError(f"Expected vsp.exe or vspaero.exe in {root} or its bin folder")

        # These are the import locations used by the original Windows runner.
        for folder in reversed([python_dir / "openvsp_config", python_dir / "openvsp", python_dir]):
            if folder.is_dir() and str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        for folder in dict.fromkeys([root, exe_dir, python_dir / "openvsp" / "openvsp"]):
            if not folder.is_dir():
                continue
            path = str(folder)
            if hasattr(os, "add_dll_directory") and path not in _DLL_HANDLES:
                _DLL_HANDLES[path] = os.add_dll_directory(path)
            if path not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    try:
        config = importlib.import_module("openvsp_config")
    except ModuleNotFoundError as error:
        if error.name != "openvsp_config":
            raise
    else:
        config._IGNORE_IMPORTS = True
    vsp = importlib.import_module("openvsp")
    if not hasattr(vsp, "VSPCheckSetup"):
        vsp = importlib.import_module("openvsp.vsp")
    vsp.VSPCheckSetup()
    return vsp


def _run_sweep(vsp, wing_id, alphas, mach, reynolds, reference, velocity, density):
    """Compute the wing mesh, then solve at one Mach and multiple alphas."""
    geometry, sweep = "VSPAEROComputeGeometry", "VSPAEROSweep"
    missing = {geometry, sweep} - set(vsp.ListAnalysis())
    if missing:
        raise RuntimeError(f"Required OpenVSP analyses unavailable: {sorted(missing)}")

    # Both stages use the same visible lifting surfaces and no thick-body set.
    def set_defaults(name):
        vsp.SetAnalysisInputDefaults(name)
        vsp.SetIntAnalysisInput(name, "GeomSet", [vsp.SET_NONE], 0)
        vsp.SetIntAnalysisInput(name, "ThinGeomSet", [vsp.SET_SHOWN], 0)

    set_defaults(geometry)
    vsp.SetIntAnalysisInput(geometry, "Symmetry", [0], 0)
    vsp.Update()
    print("VSPAERO: computing geometry...", flush=True)
    vsp.ExecAnalysis(geometry)

    set_defaults(sweep)
    vsp.SetIntAnalysisInput(sweep, "RefFlag", [vsp.MANUAL_REF], 0)
    vsp.SetIntAnalysisInput(sweep, "NCPU", [2], 0)
    vsp.SetStringAnalysisInput(sweep, "WingID", [wing_id], 0)
    xcg, ycg, zcg = reference["cg"]
    for name, value in {
        "Sref": reference["Sref"], "bref": reference["Bref"], "cref": reference["Cref"],
        "Vinf": velocity, "Rho": density,
        "Xcg": xcg, "Ycg": ycg, "Zcg": zcg, "ReCref": reynolds,
        "MachStart": mach, "MachEnd": mach,
        "AlphaStart": min(alphas), "AlphaEnd": max(alphas),
    }.items():
        vsp.SetDoubleAnalysisInput(sweep, name, [float(value)], 0)
    vsp.SetIntAnalysisInput(sweep, "MachNpts", [1], 0)
    vsp.SetIntAnalysisInput(sweep, "AlphaNpts", [len(alphas)], 0)
    vsp.Update()
    print("VSPAERO: running sweep...", flush=True)
    return vsp.ExecAnalysis(sweep)


def _read_results(vsp, result_id, alphas, notes):
    """Select final values from per-angle histories and restore input order."""
    rows = {}
    for child_id in vsp.GetStringResults(result_id, "ResultsVec"):
        available = set(vsp.GetAllDataNames(child_id))

        def last_value(*names):
            for name in names:
                if name in available:
                    values = vsp.GetDoubleResults(child_id, name)
                    if values is not None and len(values):
                        return float(values[-1])
            return None

        alpha = last_value("Alpha", "AlphaDeg")
        cl, cd = last_value("CLtot", "CL"), last_value("CDtot", "CD")
        if alpha is None or cl is None or cd is None:
            continue  # ResultsVec can also contain mesh and other non-history data.
        target = next((a for a in alphas if math.isclose(a, alpha, abs_tol=1e-8)), None)
        if target is None:
            continue
        cm = last_value("CMytot")
        row = (alpha, cl, cd, cm)
        # Preserve the original duplicate-selection rule; this is a heuristic.
        if target not in rows or abs(cl) + abs(cd) > abs(rows[target][1]) + abs(rows[target][2]):
            rows[target] = row

    output = {name: [] for name in ("Alpha", "CL", "CD", "Cm")}
    for alpha in alphas:
        if alpha not in rows:
            notes.append(f"WARNING: No aerodynamic result for alpha={alpha} degrees.")
            continue
        if rows[alpha][3] is None:
            notes.append(f"WARNING: Missing CMytot at alpha={alpha}; Cm is None.")
        for name, value in zip(output, rows[alpha]):
            output[name].append(value)
    return output


def _checked_set(vsp, geom_id, name, group, value):
    parameter = vsp.FindParm(geom_id, name, group)
    if not parameter:
        raise RuntimeError(f"Missing OpenVSP parameter {group}.{name}")
    vsp.SetParmVal(parameter, float(value))


def _add_aircraft_wing(vsp, wing):
    """Build mirrored semispan sections; twists are absolute at outboard stations."""
    sections = wing.wing_sections
    if not sections or len(wing.airfoils) != len(sections) + 1:
        raise ValueError("Each wing needs sections and one airfoil per station")
    for i, section in enumerate(sections):
        values = vars(section)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Wing section values must be finite")
        if min(section.span, section.root_chord, section.tip_chord) <= 0:
            raise ValueError("Section spans and chords must be positive")
        if not 0 <= section.twist_loc <= 1:
            raise ValueError("twist_loc must be a chord fraction in [0, 1]")
        if i and not math.isclose(sections[i-1].tip_chord, section.root_chord,
                                  rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Adjacent sections must share a chord at their common station")
    geom_id = vsp.AddGeom("WING", "")
    vsp.SetGeomName(geom_id, wing.name)
    for i in range(1, len(sections)):
        vsp.InsertXSec(geom_id, i, vsp.XS_FOUR_SERIES)
    _checked_set(vsp, geom_id, "Sym_Planar_Flag", "Sym", vsp.SYM_XZ)
    _checked_set(vsp, geom_id, "RelativeTwistFlag", "WingGeom", 0)
    _checked_set(vsp, geom_id, "Twist", "XSec_0", 0)
    _checked_set(vsp, geom_id, "Twist_Location", "XSec_0", sections[0].twist_loc)
    for i, section in enumerate(sections, 1):
        vsp.SetDriverGroup(geom_id, i, vsp.SPAN_WSECT_DRIVER,
                           vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
        for name, value in {
            "Span": section.span, "Root_Chord": section.root_chord,
            "Tip_Chord": section.tip_chord, "Twist": section.twist_deg,
            "Twist_Location": section.twist_loc, "Sweep": section.sweep_deg,
            "Sweep_Location": 0, "Dihedral": section.dihedral_deg,
        }.items():
            _checked_set(vsp, geom_id, name, f"XSec_{i}", value)
        vsp.Update()
    for i, foil in enumerate(wing.airfoils):
        if not (0 < foil.thick_chord < 1 and 0 <= foil.camber < 1
                and 0 < foil.camber_loc < 1):
            raise ValueError("Invalid four-series airfoil parameters")
        for name, value in {"ThickChord": foil.thick_chord, "Camber": foil.camber,
                            "CamberLoc": foil.camber_loc}.items():
            _checked_set(vsp, geom_id, name, f"XSecCurve_{i}", value)
    for axis in "XYZ":
        for suffix, attribute in [("Location", "loc"), ("Rotation", "rot")]:
            value = float(getattr(wing, f"{axis.lower()}_{attribute}"))
            if not math.isfinite(value):
                raise ValueError("Wing transforms must be finite")
            _checked_set(vsp, geom_id, f"{axis}_Rel_{suffix}", "XForm", value)
    vsp.Update()
    # Read back every section: OpenVSP may silently clamp or couple parameters.
    for i, section in enumerate(sections, 1):
        for name, expected in {"Span": section.span, "Root_Chord": section.root_chord,
                               "Tip_Chord": section.tip_chord, "Twist": section.twist_deg}.items():
            actual = vsp.GetParmVal(geom_id, name, f"XSec_{i}")
            if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
                raise RuntimeError(f"Section {i} {name}: requested {expected}, got {actual}")
    area = sum(s.span * (s.root_chord + s.tip_chord) for s in sections)
    span = 2 * sum(s.span for s in sections)
    for name, expected in [("TotalArea", area), ("TotalSpan", span)]:
        actual = vsp.GetParmVal(geom_id, name, "WingGeom")
        if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
            raise RuntimeError(f"{name}: requested {expected}, got {actual}")
    return geom_id, area, span


def vsp_run(aircraft, V, alpha, *, h=0.0, out_dir=None,
            vsp3_name="autotwin_aircraft.vsp3", openvsp_root=None):
    """Evaluate an Aircraft at V [m/s], alpha [deg], and optional h [m].

    Each WingXsec describes a semispan segment mirrored about the XZ plane.
    Its twist is the absolute outboard-station twist; root twist is zero.
    Mach and Reynolds number use the standard atmosphere at h and Cref=Sref/Bref.
    References are the summed wing area and largest full span. Calls are serial:
    OpenVSP owns global model state. Each call gets a unique directory by default.
    """
    from ambiance import Atmosphere
    import tempfile

    V, alpha, h = float(V), float(alpha), float(h)
    if not all(math.isfinite(x) for x in (V, alpha, h)) or V <= 0:
        raise ValueError("V must be positive; V, alpha and h must be finite")
    if not aircraft.wing_list:
        raise ValueError("Aircraft must contain at least one wing")
    atmosphere = Atmosphere(h)
    mach = V / float(atmosphere.speed_of_sound[0])
    density = float(atmosphere.density[0])
    vsp = _load_openvsp(openvsp_root or os.environ.get("OPENVSP_ROOT") or DEFAULT_OPENVSP_ROOT)
    vsp.ClearVSPModel()
    vsp.DeleteAllResults()
    wing_data = [_add_aircraft_wing(vsp, wing) for wing in aircraft.wing_list]
    wings = [item[0] for item in wing_data]
    area = sum(item[1] for item in wing_data)
    span = max(item[2] for item in wing_data)
    reference = {"Sref": area, "Bref": span, "Cref": area / span, "cg": (0, 0, 0),
                 "S_ref": area, "b_ref": span}
    reynolds = V * reference["Cref"] / float(atmosphere.kinematic_viscosity[0])
    for wing_id in wings:
        vsp.SetSetFlag(wing_id, vsp.SET_SHOWN, True)
        vsp.SetSetFlag(wing_id, vsp.SET_NOT_SHOWN, False)
    if out_dir is None:
        parent = Path("vsp_runs/optimization_evaluations").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="eval_", dir=parent))
    else:
        folder = Path(out_dir).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
    if Path(vsp3_name).name != vsp3_name:
        raise ValueError("vsp3_name must be a filename")
    model_path = folder / vsp3_name
    vsp.WriteVSPFile(str(model_path), vsp.SET_ALL)
    result_id = _run_sweep(vsp, wings[0], [alpha], mach, reynolds, reference, V, density)
    notes = []
    sweep_outputs = _read_results(vsp, result_id, [alpha], notes)
    for name in ("CL", "CD"):
        values = sweep_outputs[name]
        if len(values) != 1 or not math.isfinite(values[0]):
            raise RuntimeError(f"VSPAERO returned no finite {name}: {notes}")
    if sweep_outputs["CD"][0] < 0:
        raise RuntimeError("VSPAERO returned negative CD")
    return {
        "vsp3_path": str(model_path), "geom_ids": {"wing": wings},
        "reference": reference,
        "analysis": {"results_id": result_id, "Mach": mach, "ReCref": reynolds,
                     "V": V, "h": h, "alpha": alpha},
        "sweep_outputs": sweep_outputs, "notes": notes,
    }


# Convenient name for the optimization script; this is not a pytest test.
test_vsp_run = vsp_run
test_vsp_run.__test__ = False
