from __future__ import annotations

import os
import ast
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

_OPENVSP_DLL_HANDLES: List[Any] = []


def configure_openvsp_python(openvsp_root: str | Path) -> Dict[str, str]:
    openvsp_root = Path(openvsp_root).resolve()
    vsp_py = openvsp_root / "python"
    if not vsp_py.exists():
        raise FileNotFoundError(f"OpenVSP python folder not found: {vsp_py}")

    # Find exe dir (root in your distribution)
    candidates = [openvsp_root, openvsp_root / "bin"]
    vsp_exe_dir = None
    for c in candidates:
        if (c / "vsp.exe").exists() or (c / "vspaero.exe").exists():
            vsp_exe_dir = c
            break
    if vsp_exe_dir is None:
        for p in openvsp_root.rglob("vsp.exe"):
            vsp_exe_dir = p.parent
            break
    if vsp_exe_dir is None:
        raise FileNotFoundError(f"Could not locate vsp.exe under: {openvsp_root}")

    # Import roots (avoid namespace collisions)
    import_roots = [
        vsp_py / "openvsp_config",
        vsp_py / "openvsp",
        vsp_py,
    ]
    for p in reversed(import_roots):
        if p.exists():
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    # DLL dirs
    dll_dirs = [openvsp_root, vsp_exe_dir, vsp_py / "openvsp" / "openvsp"]
    seen = set()
    for d in dll_dirs:
        if not d.exists():
            continue
        d_str = str(d)
        if d_str in seen:
            continue
        seen.add(d_str)
        if hasattr(os, "add_dll_directory"):
            _OPENVSP_DLL_HANDLES.append(os.add_dll_directory(d_str))
        os.environ["PATH"] = d_str + os.pathsep + os.environ.get("PATH", "")

    return {"openvsp_root": str(openvsp_root), "vsp_py": str(vsp_py), "vsp_exe_dir": str(vsp_exe_dir)}


def _pick_analysis(vsp, preferred: str, fallbacks: List[str]) -> str:
    """Pick the first analysis name that exists in this OpenVSP build."""
    names = set(vsp.ListAnalysis())
    if preferred in names:
        return preferred
    for f in fallbacks:
        if f in names:
            return f
    raise RuntimeError(f"Could not find analysis '{preferred}' or fallbacks {fallbacks}. Available: {sorted(names)}")


def _force_geoms_into_set_all(vsp, geom_ids: List[str]) -> None:
    """
    Safety: ensure all geoms are in SET_ALL so they can be referenced by ThinGeomSet=SET_ALL.
    This doesn't replace analysis-set selection; it's just a guardrail.
    """
    for gid in geom_ids:
        try:
            vsp.SetSetFlag(gid, vsp.SET_ALL, True)
        except Exception:
            pass


def _set_airframe_prop_visibility(vsp, airframe_ids: List[str], prop_ids: List[str]) -> None:
    """Show airframe geoms and hide propeller geoms for airframe-only VSPAERO runs."""
    for gid in airframe_ids:
        try:
            vsp.SetSetFlag(gid, vsp.SET_SHOWN, True)
            vsp.SetSetFlag(gid, vsp.SET_NOT_SHOWN, False)
        except Exception:
            pass

    for gid in prop_ids:
        try:
            vsp.SetSetFlag(gid, vsp.SET_SHOWN, False)
            vsp.SetSetFlag(gid, vsp.SET_NOT_SHOWN, True)
        except Exception:
            pass


def _set_total_span(vsp, geom_id: str, span: float, label: str, notes: List[str]) -> None:
    for pname, group in [("TotalSpan", "WingGeom"), ("Total_Span", "Plan"), ("Span", "XSec_1")]:
        try:
            vsp.SetParmVal(geom_id, pname, group, span)
            return
        except Exception:
            pass
    notes.append(f"WARNING: Could not set {label} total span (tried TotalSpan/Total_Span/Span).")


def _get_alpha_list_deg(input_dict: Dict[str, Any], notes: List[str]) -> List[float]:
    raw_alpha = input_dict.get("alpha_list_deg")
    if raw_alpha is None:
        default_alpha = [0.0, 1.0, 2.0]
        notes.append(f"INFO: Using default for 'alpha_list_deg' = {default_alpha}.")
        return default_alpha

    if isinstance(raw_alpha, str):
        text = raw_alpha.strip()
        try:
            raw_alpha = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raw_alpha = [item.strip() for item in text.split(",") if item.strip()]

    if isinstance(raw_alpha, (int, float)) and not isinstance(raw_alpha, bool):
        alpha_list = [float(raw_alpha)]
    elif isinstance(raw_alpha, (list, tuple)):
        alpha_list = [float(alpha) for alpha in raw_alpha]
    else:
        raise ValueError(
            "'alpha_list_deg' must be a number, a list/tuple of numbers, "
            "or a list-like/comma-separated string."
        )

    if not alpha_list:
        raise ValueError("'alpha_list_deg' must contain at least one alpha value.")
    return alpha_list


def _set_fuselage_xsec_count(vsp, fuselage_id: str, target_count: int = 6, max_iters: int = 50) -> str:
    """
    Force fuselage xsec count with loop guards so OpenVSP no-op behavior cannot
    create an infinite loop.
    Returns the updated xsec surface id.
    """
    xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0)
    stable_iters = 0
    while vsp.GetNumXSec(xsec_surf_id) != target_count:
        before = vsp.GetNumXSec(xsec_surf_id)

        if before < target_count:
            # Different OpenVSP builds can behave differently by insert index.
            candidate_indices = []
            for idx in (before - 1, 0, 1, 2, 3, 4):
                if idx not in candidate_indices and idx >= 0:
                    candidate_indices.append(idx)
            for idx in candidate_indices:
                vsp.InsertXSec(fuselage_id, idx, vsp.XS_ELLIPSE)
                vsp.Update()
                xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0)
                if vsp.GetNumXSec(xsec_surf_id) > before:
                    break
        else:
            vsp.CutXSec(fuselage_id, before - 1)
            vsp.Update()
            xsec_surf_id = vsp.GetXSecSurf(fuselage_id, 0)

        after = vsp.GetNumXSec(xsec_surf_id)
        if after == before:
            stable_iters += 1
        else:
            stable_iters = 0

        if stable_iters >= max_iters:
            raise RuntimeError(
                f"Could not set fuselage XSec count to {target_count}; count remained {after} "
                f"for {stable_iters} consecutive attempts."
            )
    return xsec_surf_id


def run_vspaero_workflow(
    vsp,
    vsp3_path: Path,
    wing_id: str,
    alpha_list_deg: List[float],
    mach: float,
    reynolds_cref: float,
    Sref: float,
    Bref: float,
    Cref: float,
    cg_xyz: Tuple[float, float, float],
    notes: List[str],
) -> str:
    """
    Implements the Bertin-Smith style workflow:
    1) VSPAEROComputeGeometry (CompGeom) with GeomSet=SET_NONE, ThinGeomSet=SET_SHOWN
    2) VSPAEROSweep with same sets + WingID + AlphaStart/End/Npts + MachStart/End/Npts
    """

    compgeom_name = _pick_analysis(
        vsp,
        preferred="VSPAEROComputeGeometry",
        fallbacks=["VSPAERO Compute Geometry", "VSPAERO_Compute_Geometry", "VSPAEROComputeGeom", "CompGeom"],
    )
    sweep_name = _pick_analysis(
        vsp,
        preferred="VSPAEROSweep",
        fallbacks=["VSPAERO Sweep", "VSPAERO_Sweep"],
    )

    print("VSPAERO: configuring compute-geometry analysis...", flush=True)
    # (1) Compute geometry
    vsp.SetAnalysisInputDefaults(compgeom_name)

    try:
        vsp.SetIntAnalysisInput(compgeom_name, "GeomSet", [vsp.SET_NONE], 0)
    except Exception:
        vsp.SetIntAnalysisInput(compgeom_name, "GeomSet", [vsp.SET_NONE])

    try:
        vsp.SetIntAnalysisInput(compgeom_name, "ThinGeomSet", [vsp.SET_SHOWN], 0)
    except Exception:
        vsp.SetIntAnalysisInput(compgeom_name, "ThinGeomSet", [vsp.SET_SHOWN])

    try:
        vsp.SetIntAnalysisInput(compgeom_name, "Symmetry", [0], 0)
    except Exception:
        pass

    vsp.Update()
    print("VSPAERO: running compute-geometry analysis...", flush=True)
    compgeom_res = vsp.ExecAnalysis(compgeom_name)

    # (2) Sweep
    print("VSPAERO: configuring sweep analysis...", flush=True)
    vsp.SetAnalysisInputDefaults(sweep_name)

    try:
        vsp.SetIntAnalysisInput(sweep_name, "GeomSet", [vsp.SET_NONE], 0)
    except Exception:
        vsp.SetIntAnalysisInput(sweep_name, "GeomSet", [vsp.SET_NONE])

    try:
        vsp.SetIntAnalysisInput(sweep_name, "ThinGeomSet", [vsp.SET_SHOWN], 0)
    except Exception:
        vsp.SetIntAnalysisInput(sweep_name, "ThinGeomSet", [vsp.SET_SHOWN])

    try:
        vsp.SetStringAnalysisInput(sweep_name, "WingID", [wing_id], 0)
    except Exception:
        vsp.SetStringAnalysisInput(sweep_name, "WingID", [wing_id])

    # Reference (your build uses lowercase bref/cref)
    try:
        vsp.SetDoubleAnalysisInput(sweep_name, "Sref", [float(Sref)], 0)
        vsp.SetDoubleAnalysisInput(sweep_name, "bref", [float(Bref)], 0)
        vsp.SetDoubleAnalysisInput(sweep_name, "cref", [float(Cref)], 0)
    except Exception:
        vsp.SetDoubleAnalysisInput(sweep_name, "Sref", [float(Sref)])
        vsp.SetDoubleAnalysisInput(sweep_name, "bref", [float(Bref)])
        vsp.SetDoubleAnalysisInput(sweep_name, "cref", [float(Cref)])

    for k, val in zip(["Xcg", "Ycg", "Zcg"], cg_xyz):
        try:
            vsp.SetDoubleAnalysisInput(sweep_name, k, [float(val)], 0)
        except Exception:
            vsp.SetDoubleAnalysisInput(sweep_name, k, [float(val)])

    try:
        vsp.SetDoubleAnalysisInput(sweep_name, "ReCref", [float(reynolds_cref)], 0)
    except Exception:
        vsp.SetDoubleAnalysisInput(sweep_name, "ReCref", [float(reynolds_cref)])

    try:
        vsp.SetDoubleAnalysisInput(sweep_name, "MachStart", [float(mach)], 0)
        vsp.SetDoubleAnalysisInput(sweep_name, "MachEnd", [float(mach)], 0)
        vsp.SetIntAnalysisInput(sweep_name, "MachNpts", [1], 0)
    except Exception:
        vsp.SetDoubleAnalysisInput(sweep_name, "MachStart", [float(mach)])
        vsp.SetDoubleAnalysisInput(sweep_name, "MachEnd", [float(mach)])
        vsp.SetIntAnalysisInput(sweep_name, "MachNpts", [1])

    alpha0 = float(min(alpha_list_deg))
    alphaf = float(max(alpha_list_deg))
    npts = int(len(alpha_list_deg))
    try:
        vsp.SetDoubleAnalysisInput(sweep_name, "AlphaStart", [alpha0], 0)
        vsp.SetDoubleAnalysisInput(sweep_name, "AlphaEnd", [alphaf], 0)
        vsp.SetIntAnalysisInput(sweep_name, "AlphaNpts", [npts], 0)
    except Exception:
        vsp.SetDoubleAnalysisInput(sweep_name, "AlphaStart", [alpha0])
        vsp.SetDoubleAnalysisInput(sweep_name, "AlphaEnd", [alphaf])
        vsp.SetIntAnalysisInput(sweep_name, "AlphaNpts", [npts])

    vsp.Update()
    print("VSPAERO: running sweep analysis...", flush=True)
    rid = vsp.ExecAnalysis(sweep_name)
    return rid

# Aircraft geometry generation helper functions
def _add_fuselage_item(vsp, spec: Dict[str, Any], notes: List[str]) -> str:
    gid = vsp.AddGeom("FUSELAGE", "")
    vsp.SetGeomName(gid, str(spec.get("name", "Fuselage")))

    try:
        vsp.SetParmVal(gid, "Length", "Design", float(spec["length"]))
    except Exception as e:
        notes.append(f"WARNING: Fuselage length set failed for {spec.get('name', gid)}: {e}")

    xsec_indices = sorted(
        {
            int(match.group(1))
            for key in spec
            for match in [re.fullmatch(r"XSec_(\d+)_(?:XLoc|ZLoc|w|h|diameter|Diameter)", str(key))]
            if match
        }
    )
    target_count = int(spec.get("xsec_count", max(xsec_indices) + 1 if xsec_indices else 2))

    try:
        xsec_surf_id = _set_fuselage_xsec_count(vsp, gid, target_count=target_count, max_iters=50)
        num_xsec = vsp.GetNumXSec(xsec_surf_id)
        if num_xsec != target_count:
            raise RuntimeError(f"Expected {target_count} fuselage XSecs, got {num_xsec}.")

        shape = str(spec.get("shape", "rectangle")).lower()
        for i in range(num_xsec):
            if shape in {"circle", "circular"}:
                vsp.ChangeXSecShape(xsec_surf_id, i, vsp.XS_CIRCLE)
            else:
                vsp.ChangeXSecShape(xsec_surf_id, i, vsp.XS_EDIT_CURVE)
                xsec_id = vsp.GetXSec(xsec_surf_id, i)
                shape_pid = vsp.GetXSecParm(xsec_id, "ShapeType")
                if shape_pid:
                    vsp.SetParmVal(shape_pid, vsp.EDIT_XSEC_RECTANGLE)
                else:
                    notes.append(f"WARNING: Could not access ShapeType parm on fuselage XSec_{i}.")
                vsp.EditXSecInitShape(xsec_id)
        vsp.Update()

        for i in range(num_xsec):
            xsec_id = vsp.GetXSec(xsec_surf_id, i)
            width = spec.get(f"XSec_{i}_w")
            height = spec.get(f"XSec_{i}_h")
            diameter = spec.get(f"XSec_{i}_diameter", spec.get(f"XSec_{i}_Diameter"))

            if shape in {"circle", "circular"}:
                if diameter is None and width is not None and height is not None:
                    diameter = 0.5 * (float(width) + float(height))
                if diameter is None:
                    notes.append(f"WARNING: Missing diameter for circular fuselage XSec_{i}.")
                    continue
                circle_pid = vsp.GetXSecParm(xsec_id, "Circle_Diameter")
                if circle_pid:
                    vsp.SetParmVal(circle_pid, float(diameter))
                else:
                    notes.append(f"WARNING: Could not set circle diameter on fuselage XSec_{i}.")
            else:
                if width is None or height is None:
                    notes.append(f"WARNING: Missing width/height for fuselage XSec_{i}.")
                    continue
                try:
                    vsp.SetXSecWidthHeight(xsec_id, float(width), float(height))
                except Exception:
                    notes.append(f"WARNING: Could not set width/height on fuselage XSec_{i}.")

            for parm_name, key_suffix in [("XLocPercent", "XLoc"), ("ZLocPercent", "ZLoc")]:
                if f"XSec_{i}_{key_suffix}" not in spec:
                    continue
                value = float(spec[f"XSec_{i}_{key_suffix}"])
                parm_id = vsp.GetXSecParm(xsec_id, parm_name)
                try:
                    if parm_id:
                        vsp.SetParmVal(parm_id, value)
                    else:
                        vsp.SetParmVal(gid, parm_name, f"XSec_{i}", value)
                except Exception:
                    notes.append(f"WARNING: Could not set {parm_name} on fuselage XSec_{i}.")

            for parm_name in ["RightLAngle", "TopLAngle", "RightLStrength", "TopLStrength"]:
                section_key = f"XSec_{i}_{parm_name}"
                if section_key in spec:
                    value = float(spec[section_key])
                elif parm_name in spec:
                    value = float(spec[parm_name])
                else:
                    continue
                parm_id = vsp.GetXSecParm(xsec_id, parm_name)
                try:
                    if parm_id:
                        vsp.SetParmVal(parm_id, value)
                    else:
                        vsp.SetParmVal(gid, parm_name, f"XSec_{i}", value)
                except Exception:
                    notes.append(f"WARNING: Could not set {parm_name} on fuselage XSec_{i}.")
    
    except Exception as e:
        notes.append(f"WARNING: Fuselage xsec setup failed for {spec.get('name', gid)}: {e}")

    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", float(spec.get("fuselage_XLoc", spec.get("x", 0.0))))
    vsp.SetParmVal(gid, "Y_Rel_Location", "XForm", float(spec.get("fuselage_YLoc", spec.get("y", 0.0))))
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", float(spec.get("fuselage_ZLoc", spec.get("z", 0.0))))
    vsp.SetParmVal(gid, "X_Rel_Rotation", "XForm", float(spec.get("fuselage_XRot", spec.get("x_rot", 0.0))))
    vsp.SetParmVal(gid, "Y_Rel_Rotation", "XForm", float(spec.get("fuselage_YRot", spec.get("y_rot", 0.0))))
    vsp.SetParmVal(gid, "Z_Rel_Rotation", "XForm", float(spec.get("fuselage_ZRot", spec.get("z_rot", 0.0))))
    vsp.Update()

    return gid


def _set_vsp_parm_first(vsp, geom_id: str, names: List[str], groups: List[str], value: float) -> bool:
    for name in names:
        for group in groups:
            try:
                vsp.SetParmVal(geom_id, name, group, float(value))
                return True
            except Exception:
                pass
    return False


def _set_vsp_parm_on_targets(vsp, target_ids: List[str], names: List[str], groups: List[str], value: float) -> bool:
    for target_id in target_ids:
        for name in names:
            for group in groups:
                try:
                    vsp.SetParmVal(target_id, name, group, float(value))
                    return True
                except Exception:
                    pass
    return False


def _add_wing_control_surface(vsp, wing_id: str, spec: Dict[str, Any], notes: List[str]) -> None:
    ctrl_start = spec.get("ctrl_start")
    ctrl_end = spec.get("ctrl_end")
    ctrl_hinge = spec.get("ctrl_hinge")
    wing_name = str(spec.get("name", wing_id))

    if ctrl_start is None and ctrl_end is None and ctrl_hinge is None:
        return
    if ctrl_start is None or ctrl_end is None or ctrl_hinge is None:
        notes.append(
            f"WARNING: Skipping control surface for {wing_name}; "
            "ctrl_start, ctrl_end, and ctrl_hinge are all required."
        )
        return

    eta_start = float(ctrl_start)
    eta_end = float(ctrl_end)
    hinge = float(ctrl_hinge)

    if not (0.0 <= eta_start <= 1.0 and 0.0 <= eta_end <= 1.0 and 0.0 <= hinge <= 1.0):
        notes.append(
            f"WARNING: Skipping control surface for {wing_name}; "
            "ctrl_start, ctrl_end, and ctrl_hinge must be between 0 and 1."
        )
        return
    if eta_start >= eta_end:
        notes.append(
            f"WARNING: Skipping control surface for {wing_name}; "
            f"ctrl_start ({eta_start}) must be less than ctrl_end ({eta_end})."
        )
        return

    try:
        ctrl_id = vsp.AddSubSurf(wing_id, vsp.SS_CONTROL)
    except Exception as e:
        notes.append(f"WARNING: Could not add control surface to {wing_name}: {e}")
        return

    try:
        sub_surf_count = int(vsp.GetNumSubSurf(wing_id))
    except Exception:
        sub_surf_count = 1

    group_candidates = [
        f"SS_Control_{sub_surf_count}",
        "SS_Control_1",
        "SS_Control",
    ]
    target_ids = [wing_id, ctrl_id]

    parm_specs = [
        (["EtaFlag"], 1.0),
        (["EtaStart"], eta_start),
        (["EtaEnd"], eta_end),
        (["Length_C_Start"], hinge),
        (["Length_C_End"], float(spec.get("ctrl_hinge_end", hinge))),
        (["LE_Flag", "LEFlag"], float(spec.get("ctrl_le_flag", 0.0))),
    ]
    for parm_names, value in parm_specs:
        if not _set_vsp_parm_on_targets(vsp, target_ids, parm_names, group_candidates, value):
            notes.append(f"WARNING: Could not set {parm_names[0]} for control surface on {wing_name}.")


def _add_propeller_item(vsp, spec: Dict[str, Any], notes: List[str]) -> str:
    gid = vsp.AddGeom("PROP", "")
    name = str(spec.get("name", "Propeller"))
    vsp.SetGeomName(gid, name)

    diameter = spec.get("diameter")
    if diameter is None and spec.get("radius") is not None:
        diameter = 2.0 * float(spec["radius"])
    if diameter is not None:
        if not _set_vsp_parm_first(vsp, gid, ["Diameter"], ["Design"], float(diameter)):
            notes.append(f"WARNING: Could not set diameter for propeller {name}.")
    else:
        notes.append(f"WARNING: Missing diameter/radius for propeller {name}.")

    parm_specs = [
        (["num_blades", "NumBlade", "num_blade"], ["NumBlade"], ["Design"]),
        (["beta34", "Beta34"], ["Beta34"], ["Design"]),
        (["chord", "Chord"], ["Chord"], ["Design"]),
        (["twist", "Twist"], ["Twist"], ["Design"]),
        (["pitch", "Pitch"], ["Pitch"], ["Design"]),
        (["rotate", "Rotate"], ["Rotate"], ["Design"]),
        (["reverse", "ReverseFlag"], ["ReverseFlag"], ["Design"]),
        (["prop_mode", "PropMode"], ["PropMode"], ["Design"]),
    ]
    for input_keys, parm_names, groups in parm_specs:
        value = next((spec[key] for key in input_keys if key in spec and spec[key] is not None), None)
        if value is None:
            continue
        if not _set_vsp_parm_first(vsp, gid, parm_names, groups, float(value)):
            notes.append(f"WARNING: Could not set {parm_names[0]} for propeller {name}.")

    transform_specs = [
        ("x", "X_Rel_Location"),
        ("y", "Y_Rel_Location"),
        ("z", "Z_Rel_Location"),
        ("x_rot", "X_Rel_Rotation"),
        ("y_rot", "Y_Rel_Rotation"),
        ("z_rot", "Z_Rel_Rotation"),
    ]
    for input_key, parm_name in transform_specs:
        if input_key not in spec or spec[input_key] is None:
            continue
        if not _set_vsp_parm_first(vsp, gid, [parm_name], ["XForm"], float(spec[input_key])):
            notes.append(f"WARNING: Could not set {parm_name} for propeller {name}.")

    vsp.Update()
    return gid


def _add_trapezoid_wing(vsp, spec: Dict[str, Any], notes: List[str]) -> tuple[str, Dict[str, float]]:
    S = float(spec["S"])
    AR = float(spec["AR"])
    taper = float(spec.get("taper", 1.0))
    sweep_deg = float(spec.get("sweep_deg", 0.0))

    span = math.sqrt(S * AR)
    c_root = 2 * S / (span * (1 + taper))
    c_tip = c_root * taper
    mac = (2 / 3) * c_root * ((1 + taper + taper**2) / (1 + taper))

    gid = vsp.AddGeom("WING", "")
    vsp.SetGeomName(gid, str(spec.get("name", "Wing")))

    try:
        vsp.SetDriverGroup(
            gid,
            1,
            vsp.SPAN_WSECT_DRIVER,
            vsp.ROOTC_WSECT_DRIVER,
            vsp.TIPC_WSECT_DRIVER,
        )
    except Exception:
        pass

    # Set wing planform 
    _set_total_span(vsp, gid, span, spec.get("name", "wing"), notes)
    vsp.SetParmVal(gid, "Root_Chord", "XSec_1", c_root)
    vsp.SetParmVal(gid, "Tip_Chord", "XSec_1", c_tip)
    vsp.SetParmVal(gid, "Sweep", "XSec_1", sweep_deg)
    vsp.SetParmVal(gid, "Sweep_Location", "XSec_1", 0.0)

    # Set wing location
    vsp.SetParmVal(gid, "X_Rel_Location", "XForm", float(spec.get("x", 0.0)))
    vsp.SetParmVal(gid, "Y_Rel_Location", "XForm", float(spec.get("y", 0.0)))
    vsp.SetParmVal(gid, "Z_Rel_Location", "XForm", float(spec.get("z", 0.0)))

    # Set wing rotation
    vsp.SetParmVal(gid, "X_Rel_Rotation", "XForm", float(spec.get("x_rot", 0.0)))
    vsp.SetParmVal(gid, "Y_Rel_Rotation", "XForm", float(spec.get("y_rot", 0.0)))
    vsp.SetParmVal(gid, "Z_Rel_Rotation", "XForm", float(spec.get("z_rot", 0.0)))

    # Set control surfaces
    _add_wing_control_surface(vsp, gid, spec, notes)

    # Set airfoils
    vsp.SetParmVal(gid, "ThickChord", "XSecCurve_0", float(spec.get("root_ThickChord", 0.10)))
    vsp.SetParmVal(gid, "Camber", "XSecCurve_0", float(spec.get("root_Camber", 0.04)))
    vsp.SetParmVal(gid, "CamberLoc", "XSecCurve_0", float(spec.get("root_CamberLoc", 0.4)))
    vsp.SetParmVal(gid, "ThickChord", "XSecCurve_1", float(spec.get("tip_ThickChord", 0.10)))
    vsp.SetParmVal(gid, "Camber", "XSecCurve_1", float(spec.get("tip_Camber", 0.04)))
    vsp.SetParmVal(gid, "CamberLoc", "XSecCurve_1", float(spec.get("tip_CamberLoc", 0.4)))

    return gid, {"S": S, "span": span, "mac": mac}

def vsp_run(
    input_dict: Dict[str, Any],
    out_dir: str | Path = "vsp_runs/run_001",
    vsp3_name: str = "autotwin_aircraft.vsp3",
    mach: float = 0.076,
    reynolds_cref: float = 4.4e5,
) -> Dict[str, Any]:
    notes: List[str] = []
    fuselage_xsec_parms: List[str] = []
    out_dir = input_dict.get("out_dir", out_dir)
    vsp3_name = input_dict.get("vsp3_name", vsp3_name)

    openvsp_root = Path(r"C:\Users\Berk\OneDrive\Desktop\OpenVSP-3.47.0-win64")
    paths = configure_openvsp_python(openvsp_root)

    import importlib
    try:
        openvsp_config = importlib.import_module("openvsp_config")
        setattr(openvsp_config, "_IGNORE_IMPORTS", True)
    except Exception:
        pass

    vsp = importlib.import_module("openvsp")
    if not hasattr(vsp, "VSPCheckSetup"):
        vsp = importlib.import_module("openvsp.vsp")

    vsp.VSPCheckSetup()
    print("OpenVSP import OK:", paths)
    print("OpenVSP module:", getattr(vsp, "__file__", "unknown"))

    # -----------------------------
    # Read inputs
    # -----------------------------
    def _get_float(key: str, default: Optional[float] = None, aliases: Optional[List[str]] = None) -> float:
        aliases = aliases or []
        for candidate in [key] + aliases:
            if candidate in input_dict and input_dict[candidate] is not None:
                return float(input_dict[candidate])
        if default is not None:
            notes.append(f"INFO: Using default for '{key}' = {default}.")
            return float(default)
        raise KeyError(f"Required input missing: {key}")

    def _get_text(key: str, default: Optional[str] = None, aliases: Optional[List[str]] = None) -> str:
        aliases = aliases or []
        for candidate in [key] + aliases:
            if candidate in input_dict and input_dict[candidate] is not None:
                return str(input_dict[candidate])
        if default is not None:
            notes.append(f"INFO: Using default for '{key}' = {default}.")
            return str(default)
        raise KeyError(f"Required input missing: {key}")

    alpha_list_deg = _get_alpha_list_deg(input_dict, notes)
    
    # -----------------------------
    # Build geometry (minimal)
    # -----------------------------
    print("OpenVSP: building geometry...", flush=True)
    vsp.ClearVSPModel()

    # Fuselages
    fuselage_specs = input_dict.get("fuselages")
    fuselage_ids = []
    for spec in fuselage_specs:
        fuselage_ids.append(_add_fuselage_item(vsp, spec, notes))

    # Wings
    wing_specs = input_dict["wings"]
    wing_ids = []
    wing_refs = []
    for spec in wing_specs:
        gid, ref = _add_trapezoid_wing(vsp, spec, notes)
        wing_ids.append(gid)
        wing_refs.append(ref)

    # Propellers
    prop_ids = []
    for spec in input_dict.get("props", []):
        prop_ids.append(_add_propeller_item(vsp, spec, notes))

    primary_wing_id = wing_ids[0]
    Sref = sum(r["S"] for r in wing_refs)
    Bref = max(r["span"] for r in wing_refs)
    Cref = Sref / Bref
    wing_id = primary_wing_id

    vsp.Update()

    # Safety: ensure they’re in SET_ALL
    all_geom_ids = [*fuselage_ids, *wing_ids, *prop_ids]
    _force_geoms_into_set_all(vsp, all_geom_ids)
    # Thick fuselage/boom meshes can create non-manifold panel geometry in VSPAERO.
    # Keep them in the saved VSP model, but run the aerodynamic sweep on lifting
    # surfaces only.
    _set_airframe_prop_visibility(vsp, wing_ids, [*fuselage_ids, *prop_ids])
    if fuselage_ids or prop_ids:
        notes.append("INFO: VSPAERO sweep uses wing geometry only; fuselages/props are hidden from the analysis set.")
    vsp.Update()

    # Write vsp3
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vsp3_path = out_dir / vsp3_name
    vsp.WriteVSPFile(str(vsp3_path), vsp.SET_ALL)

    rid = run_vspaero_workflow(
        vsp=vsp,
        vsp3_path=vsp3_path,
        wing_id=wing_id,
        alpha_list_deg=alpha_list_deg,
        mach=mach,
        reynolds_cref=reynolds_cref,
        Sref=Sref,
        Bref=Bref,
        Cref=Cref,
        cg_xyz=input_dict["cg_xyz"],
        notes=notes,
    )

    # Extract per-alpha results safely (ResultsVec may contain non-alpha result IDs too)
    sweep_outputs: Dict[str, Any] = {}
    def _try_get_double(res_id: str, names: List[str]) -> Optional[List[float]]:
        try:
            available_names = set(vsp.GetAllDataNames(res_id))
        except Exception:
            available_names = set()
        for n in names:
            if available_names and n not in available_names:
                continue
            try:
                v = vsp.GetDoubleResults(res_id, n)
                if v is not None and len(v) > 0:
                    return list(v)
            except Exception:
                continue
        return None
    try:
        rid_vec = vsp.GetStringResults(rid, "ResultsVec")

        results_by_alpha: Dict[float, Tuple[float, float, float, float]] = {}

        for r in rid_vec:
            # Only keep result IDs that actually contain the sweep fields
            a = _try_get_double(r, ["Alpha", "AlphaDeg"])
            cl = _try_get_double(r, ["CLtot", "CL"])
            cd = _try_get_double(r, ["CDtot", "CD"])

            if a is None or cl is None or cd is None:
                continue  # skip non-alpha result IDs

            # Moment naming varies by build; your console shows "Cmztot"
            cm = _try_get_double(r, ["CMytot"])

            alpha_val = float(a[-1])
            alpha_key = next((float(target) for target in alpha_list_deg if math.isclose(alpha_val, float(target), abs_tol=1.0e-8)), None)
            if alpha_key is None:
                continue

            row = (alpha_val, float(cl[-1]), float(cd[-1]), float(cm[-1] if cm is not None else 0.0))
            existing = results_by_alpha.get(alpha_key)
            if existing is None or abs(row[1]) + abs(row[2]) > abs(existing[1]) + abs(existing[2]):
                results_by_alpha[alpha_key] = row

        alpha_out, cl_out, cd_out, cm_out = [], [], [], []
        for alpha_key in [float(a) for a in alpha_list_deg]:
            if alpha_key not in results_by_alpha:
                continue
            alpha_val, cl_val, cd_val, cm_val = results_by_alpha[alpha_key]
            alpha_out.append(alpha_val)
            cl_out.append(cl_val)
            cd_out.append(cd_val)
            cm_out.append(cm_val)

        sweep_outputs = {"Alpha": alpha_out, "CL": cl_out, "CD": cd_out, "Cm": cm_out}

    except Exception as e:
        notes.append(f"WARNING: Could not extract ResultsVec outputs: {e}")
    
    return {
        "vsp3_path": str(vsp3_path),
        "geom_ids": {"fuselage": fuselage_ids, "wing": wing_ids, "prop": prop_ids},
        "reference": {"Sref": Sref, "Bref": Bref, "Cref": Cref, "cg": input_dict["cg_xyz"]},
        "analysis": {"results_id": rid},
        "sweep_outputs": sweep_outputs,
        "notes": notes,
        "debug": {"fuselage_xsec_parms": fuselage_xsec_parms},
    }
