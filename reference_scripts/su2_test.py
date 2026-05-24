import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) User parameters
# -----------------------------
@dataclass
class AirframeParams:
    # Wing (simple trapezoid planform, unswept by default)
    span: float = 2.0                 # full span [m]
    chord_root: float = 0.35          # [m]
    chord_tip: float = 0.20           # [m]
    sweep_le_deg: float = 10.0        # leading-edge sweep [deg]
    dihedral_deg: float = 3.0         # [deg]
    thickness_ratio: float = 0.10     # approximate thickness for a crude "thickened plate"
    # Fuselage (simple capsule)
    fuse_length: float = 1.4          # [m]
    fuse_radius: float = 0.09         # [m]
    # Positioning
    wing_x_le: float = 0.35           # x-location of wing leading edge [m]
    wing_z: float = 0.0               # wing vertical offset [m]


@dataclass
class FlowParams:
    mach: float = 0.12
    reynolds: float = 2.0e6           # used for RANS; can be ignored for Euler
    aoa_deg_list: tuple = (-4, -2, 0, 2, 4, 6, 8, 10, 12)
    # Freestream conditions (SU2 can use non-dimensional too; here is a simple approach)
    rho: float = 1.225                # [kg/m^3]
    mu: float = 1.789e-5              # [Pa*s]
    temperature: float = 288.15       # [K]


@dataclass
class MeshParams:
    # Farfield box size (half-extends)
    farfield_x: float = 15.0          # [m] upstream/downstream extent
    farfield_y: float = 15.0          # [m]
    farfield_z: float = 15.0          # [m]
    # Mesh sizing
    lc_body: float = 0.02             # characteristic length near body [m]
    lc_far: float = 1.0               # characteristic length farfield [m]


@dataclass
class SU2Params:
    # Choose: "EULER" (fast, no BL) or "RANS" (slower, needs BL refinement ideally)
    solver: str = "EULER"
    max_iter: int = 1500
    cfl: float = 3.0
    # Turbulence model if RANS
    turb_model: str = "SA"


# -----------------------------
# 2) Helpers
# -----------------------------
def run_cmd(cmd, cwd=None):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


# -----------------------------
# 3) Geometry + mesh with Gmsh
#    - Creates a very simple "thickened" wing and a capsule fuselage
#    - Embeds in a farfield box
#    - Generates a 3D volume mesh
# -----------------------------
def gmsh_geo_script(p: AirframeParams, m: MeshParams) -> str:
    sweep = np.deg2rad(p.sweep_le_deg)
    dih = np.deg2rad(p.dihedral_deg)

    half_span = 0.5 * p.span
    # Wing tip LE x offset due to sweep
    tip_x_le = p.wing_x_le + np.tan(sweep) * half_span
    # Dihedral z offset
    tip_z = p.wing_z + np.tan(dih) * half_span

    # Wing thickness (very crude)
    t_root = p.thickness_ratio * p.chord_root
    t_tip = p.thickness_ratio * p.chord_tip

    # We'll build a "wedge-like" wing as a single volume by lofting two rectangles (root & tip)
    # Root rectangle corners (LE/TE, upper/lower)
    # Coordinates (x, y, z)
    root = {
        "LEU": (p.wing_x_le, 0.0, p.wing_z + 0.5 * t_root),
        "LEL": (p.wing_x_le, 0.0, p.wing_z - 0.5 * t_root),
        "TEU": (p.wing_x_le + p.chord_root, 0.0, p.wing_z + 0.5 * t_root),
        "TEL": (p.wing_x_le + p.chord_root, 0.0, p.wing_z - 0.5 * t_root),
    }
    tip = {
        "LEU": (tip_x_le, half_span, tip_z + 0.5 * t_tip),
        "LEL": (tip_x_le, half_span, tip_z - 0.5 * t_tip),
        "TEU": (tip_x_le + p.chord_tip, half_span, tip_z + 0.5 * t_tip),
        "TEL": (tip_x_le + p.chord_tip, half_span, tip_z - 0.5 * t_tip),
    }

    # Farfield box extents
    x_min, x_max = -m.farfield_x, m.farfield_x
    y_min, y_max = -m.farfield_y, m.farfield_y
    z_min, z_max = -m.farfield_z, m.farfield_z

    # Fuselage capsule axis aligned with x
    # We'll approximate with a cylinder + two spheres (boolean union)
    # Place fuselage roughly centered around x ~ 0.6
    fuse_x0 = 0.2
    fuse_x1 = fuse_x0 + p.fuse_length

    # Gmsh .geo (OpenCASCADE) script
    return f"""
SetFactory("OpenCASCADE");

// Mesh sizes
lc_body = {m.lc_body};
lc_far  = {m.lc_far};

// ----------------------------
// Wing (loft two rectangles to a volume)
// ----------------------------
Point(1) = {{{root["LEL"][0]}, {root["LEL"][1]}, {root["LEL"][2]}, lc_body}};
Point(2) = {{{root["TEL"][0]}, {root["TEL"][1]}, {root["TEL"][2]}, lc_body}};
Point(3) = {{{root["TEU"][0]}, {root["TEU"][1]}, {root["TEU"][2]}, lc_body}};
Point(4) = {{{root["LEU"][0]}, {root["LEU"][1]}, {root["LEU"][2]}, lc_body}};

Point(5) = {{{tip["LEL"][0]}, {tip["LEL"][1]}, {tip["LEL"][2]}, lc_body}};
Point(6) = {{{tip["TEL"][0]}, {tip["TEL"][1]}, {tip["TEL"][2]}, lc_body}};
Point(7) = {{{tip["TEU"][0]}, {tip["TEU"][1]}, {tip["TEU"][2]}, lc_body}};
Point(8) = {{{tip["LEU"][0]}, {tip["LEU"][1]}, {tip["LEU"][2]}, lc_body}};

// Root rectangle
Line(1) = {{1,2}};
Line(2) = {{2,3}};
Line(3) = {{3,4}};
Line(4) = {{4,1}};
Curve Loop(1) = {{1,2,3,4}};
Plane Surface(1) = {{1}};

// Tip rectangle
Line(5) = {{5,6}};
Line(6) = {{6,7}};
Line(7) = {{7,8}};
Line(8) = {{8,5}};
Curve Loop(2) = {{5,6,7,8}};
Plane Surface(2) = {{2}};

// Loft surfaces to form wing volume
wing_surfs[] = {{1,2}};
wing_shell[] = ThruSections{{ wing_surfs[] }};
wing_vol[] = Volume{{ wing_shell[] }};

// Mirror wing to negative y to get full wing
wing_vol_m[] = Symmetry{{0,1,0,0}}{{ Volume{{wing_vol[]}}; }};

// ----------------------------
// Fuselage capsule (cylinder + spheres)
// ----------------------------
cyl = Cylinder(100) = {{{fuse_x0},0,0, {p.fuse_length},0,0, {p.fuse_radius}}};
s1  = Sphere(101)   = {{{fuse_x0},0,0, {p.fuse_radius}}};
s2  = Sphere(102)   = {{{fuse_x1},0,0, {p.fuse_radius}}};
fuse[] = BooleanUnion{{ Volume{{cyl}}; Delete; }}{{ Volume{{s1,s2}}; Delete; }};

// ----------------------------
// Combine wing + fuselage as "body"
// ----------------------------
body[] = BooleanUnion{{ Volume{{wing_vol[] , wing_vol_m[]}}; Delete; }}{{ Volume{{fuse[]}}; Delete; }};

// ----------------------------
// Farfield box and fluid volume = box - body
// ----------------------------
box = Box(200) = {{{x_min},{y_min},{z_min}, {x_max-x_min},{y_max-y_min},{z_max-z_min}}};
fluid[] = BooleanDifference{{ Volume{{box}}; Delete; }}{{ Volume{{body[]}}; }};

// ----------------------------
// Physical groups for SU2
// ----------------------------
// Mark farfield boundary surfaces:
farfield_surfs[] = Boundary{{ Volume{{fluid[]}}; }};
Physical Surface("FARFIELD") = {{ farfield_surfs[] }};

// Mark body surfaces (wing+fuselage). They are the boundary of 'body' volume:
body_surfs[] = Boundary{{ Volume{{body[]}}; }};
Physical Surface("WALL") = {{ body_surfs[] }};

Physical Volume("FLUID") = {{ fluid[] }};

// 3D mesh
Mesh.CharacteristicLengthMin = lc_body;
Mesh.CharacteristicLengthMax = lc_far;
Mesh.Algorithm3D = 10; // Delaunay
Mesh 3;
"""


def generate_mesh_with_gmsh(workdir: Path, p: AirframeParams, m: MeshParams) -> Path:
    geo_path = workdir / "airframe.geo"
    msh_path = workdir / "airframe.msh"
    su2_mesh_path = workdir / "airframe.su2"

    write_text(geo_path, gmsh_geo_script(p, m))

    # Generate .msh (v4)
    run_cmd(["gmsh", "-3", str(geo_path), "-format", "msh4", "-o", str(msh_path)], cwd=workdir)

    # Convert to SU2 mesh (SU2 has a converter, but depending on install it might be "SU2_CFD" only)
    # Many users use "SU2_GMSH2SU2" if provided. If not available, use "gmshToSU2" equivalent.
    # We'll try a couple common names.
    converters = ["SU2_GMSH2SU2", "SU2_MSH2SU2", "SU2_CFD"]
    converter_found = None
    for c in converters:
        if shutil.which(c) is not None:
            converter_found = c
            break

    if converter_found is None:
        raise RuntimeError("No SU2 converter found on PATH (tried SU2_GMSH2SU2, SU2_MSH2SU2, SU2_CFD).")

    if converter_found == "SU2_GMSH2SU2":
        run_cmd(["SU2_GMSH2SU2", str(msh_path), str(su2_mesh_path)], cwd=workdir)
    elif converter_found == "SU2_MSH2SU2":
        run_cmd(["SU2_MSH2SU2", str(msh_path), str(su2_mesh_path)], cwd=workdir)
    else:
        # Fallback: SU2_CFD cannot convert meshes; raise with guidance.
        raise RuntimeError(
            "Found SU2_CFD but no mesh converter. Install SU2 utilities (SU2_GMSH2SU2) or use a mesh format SU2 reads."
        )

    return su2_mesh_path


# -----------------------------
# 4) SU2 config writer
# -----------------------------
def write_su2_cfg(
    workdir: Path,
    mesh_su2: Path,
    flow: FlowParams,
    su2: SU2Params,
    aoa_deg: float,
    ref_area: float,
    ref_length: float,
) -> Path:
    cfg_path = workdir / f"run_AoA_{aoa_deg:+05.1f}.cfg"

    solver = su2.solver.upper().strip()
    if solver not in ("EULER", "RANS"):
        raise ValueError("SU2Params.solver must be 'EULER' or 'RANS'")

    # For 3D external aero, a typical setup:
    # - FARFIELD marker for boundaries
    # - WALL marker for body
    # Note: For RANS you should create proper BL mesh; this starter example will run but accuracy may be poor.
    cfg = f"""
% ---------------- SU2 Configuration File ----------------
SOLVER= {solver}
KIND_TURB_MODEL= {su2.turb_model if solver == "RANS" else "NONE"}

MACH_NUMBER= {flow.mach}
AoA= {aoa_deg}

FREESTREAM_TEMPERATURE= {flow.temperature}
FREESTREAM_DENSITY= {flow.rho}

% Viscosity (only used if NAVIER_STOKES / RANS)
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {flow.mu}

REF_AREA= {ref_area}
REF_LENGTH= {ref_length}
REF_ORIGIN_MOMENT_X= 0.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0

% --- Mesh ---
MESH_FILENAME= {mesh_su2.name}
MESH_FORMAT= SU2

% --- Markers ---
MARKER_FAR= ( FARFIELD )
MARKER_EULER= ( WALL )
MARKER_HEATFLUX= ( WALL, 0.0 )

% --- Numerics ---
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= {su2.cfl}
CFL_ADAPT= NO

% Convergence & iterations
ITER= {su2.max_iter}
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 50

% Output
OUTPUT_WRT_FREQ= 100
SCREEN_WRT_FREQ= 50
HISTORY_WRT_FREQ= 1
VOLUME_OUTPUT= NO
SURFACE_OUTPUT= YES
WRT_SURFACE_FORCE= YES

% Force coefficients to history
HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)

% Files
SOLUTION_FILENAME= restart.dat
RESTART_SOL= NO
"""
    write_text(cfg_path, cfg.strip() + "\n")
    return cfg_path


# -----------------------------
# 5) Run SU2 and parse results
# -----------------------------
def run_su2_case(workdir: Path, cfg_path: Path):
    # SU2_CFD is the usual solver binary
    if shutil.which("SU2_CFD") is None:
        raise RuntimeError("SU2_CFD not found on PATH. Make sure SU2 is installed and available.")
    run_cmd(["SU2_CFD", str(cfg_path.name)], cwd=workdir)


def parse_su2_history(workdir: Path) -> dict:
    # SU2 typically outputs "history.csv" or "history.dat" depending on version/config
    # We'll search for likely files:
    candidates = ["history.csv", "history.dat", "history"]
    hist_path = None
    for c in candidates:
        p = workdir / c
        if p.exists():
            hist_path = p
            break

    if hist_path is None:
        # fallback: find any file that looks like history
        for p in workdir.glob("history*"):
            hist_path = p
            break

    if hist_path is None:
        raise RuntimeError(f"No SU2 history file found in {workdir}")

    text = hist_path.read_text(errors="ignore")

    # Try to extract last reported coefficients from the file.
    # Many SU2 history outputs contain columns like: "CL", "CD", "CMx", ...
    # We'll parse using a robust "last numeric row" approach when CSV-like.
    # If it's .csv, prefer pandas.
    if hist_path.suffix.lower() == ".csv":
        df = pd.read_csv(hist_path)
        last = df.iloc[-1].to_dict()
        # Common keys
        cl = last.get("CL") or last.get("CLift") or last.get("CLIFT")
        cd = last.get("CD") or last.get("CDrag") or last.get("CDRAG")
        return {"CL": float(cl), "CD": float(cd), "history_file": hist_path.name}

    # Otherwise parse last line with many floats
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Find last line containing at least 5 numbers
    num_re = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
    best = None
    for ln in reversed(lines):
        nums = num_re.findall(ln)
        if len(nums) >= 6:
            best = ln
            break
    if best is None:
        raise RuntimeError(f"Couldn't parse coefficients from {hist_path.name}")

    # As a final fallback, try to find "CL" and "CD" tokens anywhere:
    # (Not guaranteed, but often present.)
    cl_m = re.search(r"\bCL\b[^0-9\-+]*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", text)
    cd_m = re.search(r"\bCD\b[^0-9\-+]*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", text)
    if cl_m and cd_m:
        return {"CL": float(cl_m.group(1)), "CD": float(cd_m.group(1)), "history_file": hist_path.name}

    raise RuntimeError(f"Parsed a numeric line but couldn't map it to CL/CD in {hist_path.name}.")


# -----------------------------
# 6) Main pipeline
# -----------------------------
def main():
    workdir = Path.cwd()
    p = AirframeParams()
    flow = FlowParams()
    meshp = MeshParams()
    su2 = SU2Params()

    # Reference geometry for coefficients (use wing planform area as simple Sref)
    Sref = 0.5 * p.span * (p.chord_root + p.chord_tip)  # trapezoid wing area
    cref = (2.0 / 3.0) * p.chord_root * (1 + (p.chord_tip / p.chord_root) + (p.chord_tip / p.chord_root) ** 2) / (
        1 + (p.chord_tip / p.chord_root)
    )

    print(f"Sref = {Sref:.4f} m^2, cref ≈ {cref:.4f} m")

    # 1) Mesh
    mesh_su2 = generate_mesh_with_gmsh(workdir, p, meshp)

    # 2) AoA sweep
    results = []
    for aoa in flow.aoa_deg_list:
        case_dir = workdir / f"case_AoA_{aoa:+05.1f}"
        case_dir.mkdir(exist_ok=True)

        # Copy mesh into case dir
        shutil.copy(mesh_su2, case_dir / mesh_su2.name)

        # Write config
        cfg = write_su2_cfg(
            case_dir,
            mesh_su2=case_dir / mesh_su2.name,
            flow=flow,
            su2=su2,
            aoa_deg=float(aoa),
            ref_area=Sref,
            ref_length=cref,
        )

        # Run
        run_su2_case(case_dir, cfg)

        # Parse
        coeff = parse_su2_history(case_dir)
        coeff["AoA_deg"] = float(aoa)
        results.append(coeff)

    df = pd.DataFrame(results).sort_values("AoA_deg")
    df.to_csv(workdir / "drag_polar.csv", index=False)
    print(df)

    # 3) Plot polar
    plt.figure()
    plt.plot(df["CL"], df["CD"], marker="o")
    plt.xlabel("CL")
    plt.ylabel("CD")
    plt.title("Drag Polar")
    plt.grid(True)
    plt.savefig(workdir / "drag_polar_CL_CD.png", dpi=200)

    plt.figure()
    plt.plot(df["AoA_deg"], df["CL"], marker="o")
    plt.xlabel("AoA [deg]")
    plt.ylabel("CL")
    plt.title("Lift Curve")
    plt.grid(True)
    plt.savefig(workdir / "lift_curve.png", dpi=200)

    plt.figure()
    plt.plot(df["AoA_deg"], df["CD"], marker="o")
    plt.xlabel("AoA [deg]")
    plt.ylabel("CD")
    plt.title("Drag vs AoA")
    plt.grid(True)
    plt.savefig(workdir / "drag_vs_aoa.png", dpi=200)

    print("\nWrote:")
    print(" - drag_polar.csv")
    print(" - drag_polar_CL_CD.png")
    print(" - lift_curve.png")
    print(" - drag_vs_aoa.png")


if __name__ == "__main__":
    main()
