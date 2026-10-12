"""Readable, standalone version of the OpenVSP runner (not a pytest test).

Call ``vsp_run(input_dict)`` with ``wings`` and ``cg_xyz``; ``fuselages`` and
``props`` are optional. Geometry dictionaries use the production runner's keys.
Edit DEFAULT_OPENVSP_ROOT below to change the default installation directory.
``input_dict['openvsp_root']`` or OPENVSP_ROOT can override that default.

The saved model contains every component. The aerodynamic sweep includes wings
and tails only, at one Mach number and evenly spaced angles of attack.
Importing this file does not load OpenVSP or run an analysis.
"""

from __future__ import annotations

import ast
import importlib
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


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


def _read_alphas(value):
    """Accept the numeric/list/string formats used by campaign inputs."""
    if value is None:
        value = [0.0, 1.0, 2.0]
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("alpha_list_deg must contain one or more numbers")
    alphas = [float(alpha) for alpha in value]
    if not all(math.isfinite(alpha) for alpha in alphas):
        raise ValueError("alpha_list_deg must contain finite numbers")

    # VSPAERO takes start/end/count, not an arbitrary list of angles.
    ordered = sorted(alphas)
    if len(ordered) > 1:
        step = (ordered[-1] - ordered[0]) / (len(ordered) - 1)
        if step == 0 or any(
            not math.isclose(alpha, ordered[0] + i * step, abs_tol=1e-8)
            for i, alpha in enumerate(ordered)
        ):
            raise ValueError("alpha_list_deg must be distinct and evenly spaced")
    return alphas


def _set_parameter(vsp, targets, names, groups, value, notes):
    """Find a real parameter before setting it; some VSP failures do not raise."""
    for target in targets:
        for name in names:
            for group in groups:
                parameter = vsp.FindParm(target, name, group)
                if parameter:
                    vsp.SetParmVal(parameter, float(value))
                    return
    notes.append(f"WARNING: Could not find {names[0]} on {targets[0]}.")


def _set_transform(vsp, geom_id, spec, *, fuselage=False):
    """Apply position and rotation, including the fuselage-specific input keys."""
    for axis in "XYZ":
        for suffix, input_suffix, legacy_suffix in (
            ("Location", "", "Loc"), ("Rotation", "_rot", "Rot")
        ):
            value = spec.get(axis.lower() + input_suffix, 0.0)
            if fuselage:
                value = spec.get(f"fuselage_{axis}{legacy_suffix}", value)
            vsp.SetParmVal(geom_id, f"{axis}_Rel_{suffix}", "XForm", float(value))


def _resize_fuselage(vsp, geom_id, count):
    """Adjust section count, failing if OpenVSP makes no progress."""
    if count < 2:
        raise ValueError("A fuselage needs at least two cross sections")
    surface = vsp.GetXSecSurf(geom_id, 0)
    while vsp.GetNumXSec(surface) != count:
        before = vsp.GetNumXSec(surface)
        if before < count:
            for index in dict.fromkeys([before - 1, *range(before)]):
                vsp.InsertXSec(geom_id, index, vsp.XS_ELLIPSE)
                vsp.Update()
                surface = vsp.GetXSecSurf(geom_id, 0)
                if vsp.GetNumXSec(surface) > before:
                    break
        else:
            vsp.CutXSec(geom_id, before - 1)
            vsp.Update()
            surface = vsp.GetXSecSurf(geom_id, 0)
        if abs(vsp.GetNumXSec(surface) - count) >= abs(before - count):
            raise RuntimeError(f"Could not resize fuselage from {before} to {count} sections")
    return surface


def _add_fuselage(vsp, spec, notes):
    """Build a rectangular fuselage or circular boom from its section inputs."""
    geom_id = vsp.AddGeom("FUSELAGE", "")
    vsp.SetGeomName(geom_id, str(spec.get("name", "Fuselage")))
    vsp.SetParmVal(geom_id, "Length", "Design", float(spec["length"]))
    indices = [
        int(match.group(1)) for key in spec
        if (match := re.fullmatch(r"XSec_(\d+)_(?:XLoc|ZLoc|w|h|diameter|Diameter)", str(key)))
    ]
    count = int(spec.get("xsec_count", max(indices, default=1) + 1))
    surface = _resize_fuselage(vsp, geom_id, count)
    circular = str(spec.get("shape", "rectangle")).lower() in {"circle", "circular"}

    for i in range(count):
        vsp.ChangeXSecShape(surface, i, vsp.XS_CIRCLE if circular else vsp.XS_EDIT_CURVE)
        if not circular:
            section = vsp.GetXSec(surface, i)
            parameter = vsp.GetXSecParm(section, "ShapeType")
            if not parameter:
                raise RuntimeError(f"Missing ShapeType for fuselage section {i}")
            vsp.SetParmVal(parameter, vsp.EDIT_XSEC_RECTANGLE)
            vsp.EditXSecInitShape(section)
    vsp.Update()

    for i in range(count):
        section = vsp.GetXSec(surface, i)
        prefix = f"XSec_{i}_"
        width, height = spec.get(prefix + "w"), spec.get(prefix + "h")
        if circular:
            diameter = spec.get(prefix + "diameter", spec.get(prefix + "Diameter"))
            if diameter is None and width is not None and height is not None:
                diameter = (float(width) + float(height)) / 2
            if diameter is None:
                raise ValueError(f"Missing diameter for fuselage section {i}")
            parameter = vsp.GetXSecParm(section, "Circle_Diameter")
            if not parameter:
                raise RuntimeError(f"Missing Circle_Diameter for fuselage section {i}")
            vsp.SetParmVal(parameter, float(diameter))
        else:
            if width is None or height is None:
                raise ValueError(f"Missing width/height for fuselage section {i}")
            vsp.SetXSecWidthHeight(section, float(width), float(height))

        section_values = {
            "XLocPercent": spec.get(prefix + "XLoc"),
            "ZLocPercent": spec.get(prefix + "ZLoc"),
        }
        for name in ("RightLAngle", "TopLAngle", "RightLStrength", "TopLStrength"):
            section_values[name] = spec.get(prefix + name, spec.get(name))
        for name, value in section_values.items():
            if value is None:
                continue
            parameter = vsp.GetXSecParm(section, name)
            if parameter:
                vsp.SetParmVal(parameter, float(value))
            else:
                _set_parameter(vsp, [geom_id], [name], [f"XSec_{i}"], value, notes)
    _set_transform(vsp, geom_id, spec, fuselage=True)
    vsp.Update()
    return geom_id


def _add_control_surface(vsp, wing_id, spec, notes):
    values = [spec.get(key) for key in ("ctrl_start", "ctrl_end", "ctrl_hinge")]
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("Control surfaces require ctrl_start, ctrl_end, and ctrl_hinge")
    start, end, hinge = map(float, values)
    hinge_end = float(spec.get("ctrl_hinge_end", hinge))
    if not (0 <= start < end <= 1 and 0 <= hinge <= 1 and 0 <= hinge_end <= 1):
        raise ValueError("Invalid control surface extent or chord fraction")
    control_id = vsp.AddSubSurf(wing_id, vsp.SS_CONTROL)
    groups = [f"SS_Control_{vsp.GetNumSubSurf(wing_id)}", "SS_Control_1", "SS_Control"]
    parameters = {
        "EtaFlag": 1, "EtaStart": start, "EtaEnd": end,
        "Length_C_Start": hinge, "Length_C_End": hinge_end,
        "LE_Flag": spec.get("ctrl_le_flag", 0),
    }
    for name, value in parameters.items():
        names = [name, "LEFlag"] if name == "LE_Flag" else [name]
        _set_parameter(vsp, [wing_id, control_id], names, groups, value, notes)


def _add_wing(vsp, spec, notes):
    """Convert area/aspect ratio/taper into a single trapezoidal wing."""
    area, aspect_ratio = float(spec["S"]), float(spec["AR"])
    taper = float(spec.get("taper", 1.0))
    if not (area > 0 and aspect_ratio > 0 and taper >= 0):
        raise ValueError("Wing S and AR must be positive; taper must be nonnegative")
    span = math.sqrt(area * aspect_ratio)
    root_chord = 2 * area / (span * (1 + taper))
    geom_id = vsp.AddGeom("WING", "")
    vsp.SetGeomName(geom_id, str(spec.get("name", "Wing")))
    vsp.SetDriverGroup(geom_id, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    vsp.SetParmVal(geom_id, "TotalSpan", "WingGeom", span)
    for name, value in {
        "Root_Chord": root_chord, "Tip_Chord": root_chord * taper,
        "Sweep": spec.get("sweep_deg", 0.0), "Sweep_Location": 0.0,
    }.items():
        vsp.SetParmVal(geom_id, name, "XSec_1", float(value))
    _set_transform(vsp, geom_id, spec)
    _add_control_surface(vsp, geom_id, spec, notes)
    for index, end in enumerate(("root", "tip")):
        for name, default in {"ThickChord": 0.10, "Camber": 0.04, "CamberLoc": 0.4}.items():
            vsp.SetParmVal(geom_id, name, f"XSecCurve_{index}", float(spec.get(f"{end}_{name}", default)))
    return geom_id, area, span


def _add_propeller(vsp, spec, notes):
    geom_id = vsp.AddGeom("PROP", "")
    vsp.SetGeomName(geom_id, str(spec.get("name", "Propeller")))
    diameter = spec.get("diameter")
    if diameter is None and spec.get("radius") is not None:
        diameter = 2 * float(spec["radius"])
    if diameter is None:
        raise ValueError("Propellers require diameter or radius")
    _set_parameter(vsp, [geom_id], ["Diameter"], ["Design"], diameter, notes)
    for name, keys in {
        "NumBlade": ("num_blades", "NumBlade", "num_blade"),
        "Beta34": ("beta34", "Beta34"), "Chord": ("chord", "Chord"),
        "Twist": ("twist", "Twist"), "Pitch": ("pitch", "Pitch"),
        "Rotate": ("rotate", "Rotate"), "ReverseFlag": ("reverse", "ReverseFlag"),
        "PropMode": ("prop_mode", "PropMode"),
    }.items():
        value = next((spec[key] for key in keys if spec.get(key) is not None), None)
        if value is not None:
            _set_parameter(vsp, [geom_id], [name], ["Design"], value, notes)
    _set_transform(vsp, geom_id, spec)
    vsp.Update()
    return geom_id


def _run_sweep(vsp, wing_id, alphas, mach, reynolds, reference):
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
    vsp.SetStringAnalysisInput(sweep, "WingID", [wing_id], 0)
    xcg, ycg, zcg = reference["cg"]
    for name, value in {
        "Sref": reference["Sref"], "bref": reference["Bref"], "cref": reference["Cref"],
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


def vsp_run(
    input_dict: dict[str, Any],
    out_dir: str | Path = "vsp_runs/run_001",
    vsp3_name: str = "autotwin_aircraft.vsp3",
    mach: float = 0.076,
    reynolds_cref: float = 4.4e5,
) -> dict[str, Any]:
    """Build model -> select wings -> save model -> run sweep -> read results.

    The reference convention is retained from the original: sum of all wing
    areas, largest wing span, and Cref = Sref / Bref. Mach and Reynolds remain
    function arguments. The dictionary can override out_dir and vsp3_name.
    """
    notes = []
    wing_specs = input_dict["wings"]
    if not wing_specs:
        raise ValueError("At least one wing is required")
    cg = tuple(float(value) for value in input_dict["cg_xyz"])
    if len(cg) != 3 or not all(math.isfinite(value) for value in cg):
        raise ValueError("cg_xyz must contain three finite coordinates")
    alphas = _read_alphas(input_dict.get("alpha_list_deg"))
    if input_dict.get("alpha_list_deg") is None:
        notes.append("INFO: Using default alpha_list_deg = [0.0, 1.0, 2.0].")
    openvsp_root = (
        input_dict.get("openvsp_root")
        or os.environ.get("OPENVSP_ROOT")
        or DEFAULT_OPENVSP_ROOT
    )
    vsp = _load_openvsp(openvsp_root)

    # 1. Build all components in a fresh model.
    print("OpenVSP: building geometry...", flush=True)
    vsp.ClearVSPModel()
    fuselages = [_add_fuselage(vsp, spec, notes) for spec in input_dict.get("fuselages", [])]
    wing_data = [_add_wing(vsp, spec, notes) for spec in wing_specs]
    wings = [geom_id for geom_id, _, _ in wing_data]
    props = [_add_propeller(vsp, spec, notes) for spec in input_dict.get("props", [])]
    area = sum(area for _, area, _ in wing_data)
    span = max(span for _, _, span in wing_data)
    reference = {"Sref": area, "Bref": span, "Cref": area / span, "cg": cg}

    # 2. Keep fuselages/props in the model, but exclude them from aerodynamics.
    for geom_id in [*wings, *fuselages, *props]:
        shown = geom_id in wings
        vsp.SetSetFlag(geom_id, vsp.SET_SHOWN, shown)
        vsp.SetSetFlag(geom_id, vsp.SET_NOT_SHOWN, not shown)
    if fuselages or props:
        notes.append("INFO: Sweep uses wings only; fuselages and props are hidden.")
    vsp.Update()

    # 3. Save every component before running the two analysis stages.
    folder = Path(input_dict.get("out_dir", out_dir)).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / input_dict.get("vsp3_name", vsp3_name)
    vsp.WriteVSPFile(str(model_path), vsp.SET_ALL)
    result_id = _run_sweep(vsp, wings[0], alphas, mach, reynolds_cref, reference)

    # 4. Return the campaign-facing outputs, including useful warnings.
    return {
        "vsp3_path": str(model_path),
        "geom_ids": {"fuselage": fuselages, "wing": wings, "prop": props},
        "reference": reference,
        "analysis": {"results_id": result_id},
        "sweep_outputs": _read_results(vsp, result_id, alphas, notes),
        "notes": notes,
        "debug": {},  # Campaign YAML files still declare this output.
    }
