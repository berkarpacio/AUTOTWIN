"""Pymoo + VSPAERO wing shape optimization (run from the repository root).

Examples::

    python reference_scripts/aero_shape_optim_vsp.py --check
    python reference_scripts/aero_shape_optim_vsp.py --population 20 --generations 30
    python reference_scripts/aero_shape_optim_vsp.py --openvsp-root C:/Tools/OpenVSP
    python reference_scripts/aero_shape_optim_vsp.py --patience 8
    python reference_scripts/aero_shape_optim_vsp.py --report vsp_runs/shape_optimization/RUN_FOLDER

Requires numpy, pymoo >= 0.6 and a matching Python/OpenVSP binary installation.
Uses vsp_run from the configurable, validated vsp_run_local module. Set
OPENVSP_ROOT or pass --openvsp-root when using another installation.

Like the notebook's shape example, minimize drag at prescribed CL with fixed
area and span. This runner builds one trapezoid, so the design variables are
taper, leading-edge sweep (degrees), and alpha (degrees), not 16 independent
chords. Bounds enforce positive, nonincreasing chord. Airfoils are fixed NACA
0010 sections. The lift equality is relaxed to abs(CL - target) <= tolerance.

This is an untrimmed lifting-surface study: no stall, structural sizing, or
complete aircraft parasite-drag model. CD is the runner's returned coefficient,
not an independently validated total aircraft drag prediction. Assess mesh
convergence and higher-fidelity aerodynamics before using an optimum in design.

Each invocation creates a new output directory; evaluations run serially because
OpenVSP has global model state. Baseline failure aborts immediately; subsequent
solver failures are logged and penalized. --check runs only the real baseline.
Every completed generation refreshes convergence.png/csv. Best feasible inputs
are saved after every evaluation. Ctrl+C exports completed history. --patience
optionally stops on stagnant best feasible CD; this does not prove optimality.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from libraries.aerospace.aerodynamics.vsp_run_local import vsp_run


@dataclass(frozen=True)
class Settings:
    area: float = 0.25
    span: float = 2.0
    target_cl: float = 1.0
    cl_tolerance: float = 0.01
    mach: float = 0.076
    reynolds: float = 4.4e5  # Based on Cref = area / span, as in vsp_run.
    openvsp_root: str | None = None

    def __post_init__(self):
        for name in ("area", "span", "target_cl", "cl_tolerance", "reynolds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.mach) or not 0 < self.mach < 1:
            raise ValueError("mach must be between 0 and 1")
        if self.cl_tolerance >= self.target_cl:
            raise ValueError("cl_tolerance must be smaller than target_cl")


# Edit these bounds to change the design space. Wing twist and multiple chord
# stations require extensions to the runner's geometry builder first.
VARIABLES = ("taper", "sweep_deg", "alpha_deg")
LOWER = np.array([0.15, -10.0, 0.0])
UPPER = np.array([1.0, 20.0, 15.0])
BASELINE = np.array([0.5, 0.0, 10.0])


def build_inputs(x, settings):
    taper, sweep, alpha = map(float, x)
    wing = {
        "name": "Optimized wing", "S": settings.area,
        "AR": settings.span ** 2 / settings.area,
        "taper": taper, "sweep_deg": sweep,
        "root_ThickChord": 0.10, "tip_ThickChord": 0.10,
        "root_Camber": 0.0, "tip_Camber": 0.0,
        "root_CamberLoc": 0.4, "tip_CamberLoc": 0.4,
    }
    inputs = {"wings": [wing], "fuselages": [], "props": [],
              "cg_xyz": [0.25 * settings.area / settings.span, 0, 0],
              "alpha_list_deg": [alpha]}
    if settings.openvsp_root:
        inputs["openvsp_root"] = settings.openvsp_root
    return inputs


def read_coefficients(result, alpha):
    sweep = result["sweep_outputs"]
    values = []
    for name in ("Alpha", "CL", "CD"):
        column = sweep[name]
        if len(column) != 1 or not math.isfinite(float(column[0])):
            raise ValueError(f"Invalid VSPAERO {name}: {column}")
        values.append(float(column[0]))
    actual_alpha, cl, cd = values
    if not math.isclose(actual_alpha, alpha, abs_tol=1e-6):
        raise ValueError(f"Requested alpha={alpha}, received {actual_alpha}")
    if cd <= 0:
        raise ValueError(f"Nonpositive drag coefficient: {cd}")
    return cl, cd


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def save_progress(folder, records, status="running"):
    """Export completed evaluations, including histories from interrupted old runs."""
    folder = Path(folder)
    rows = []
    best = None
    min_cv = math.inf
    for i, record in enumerate(records, 1):
        if record["status"] == "ok":
            min_cv = min(min_cv, max(0.0, record["constraint"]))
        if record.get("feasible") and (best is None or record["CD"] < best["CD"]):
            best = record
        rows.append({"evaluation": i, "CD": record.get("CD"), "CL": record.get("CL"),
                     "feasible": record.get("feasible", False),
                     "best_feasible_CD": best["CD"] if best else None,
                     "minimum_violation": min_cv if math.isfinite(min_cv) else None})
    with (folder / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evaluation", "CD", "CL", "feasible",
                                                  "best_feasible_CD", "minimum_violation"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder / "summary.json", {"status": status, "evaluations": len(rows), "best": best})
    if best:
        write_json(folder / "best_design.json", best)
        source = Path(best["directory"]) / "inputs.json"
        if source.exists():
            write_json(folder / "best_inputs.json", json.loads(source.read_text(encoding="utf-8")))
    # An explicit noninteractive canvas works in terminals and headless runs.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9, 7), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    for feasible, color, label in ((False, "0.7", "Infeasible candidate"), (True, "tab:blue", "Feasible candidate")):
        selected = [r for r in rows if r["feasible"] == feasible and r["CD"] is not None]
        axes[0].scatter([r["evaluation"] for r in selected], [r["CD"] for r in selected],
                        s=12, color=color, label=label)
    axes[0].step([r["evaluation"] for r in rows],
                 [r["best_feasible_CD"] if r["best_feasible_CD"] is not None else np.nan for r in rows],
                 where="post", color="tab:red", label="Best feasible CD so far")
    axes[0].set_ylabel("CD (lower is better at required CL)")
    axes[0].legend()
    axes[1].plot([r["evaluation"] for r in rows],
                 [r["minimum_violation"] for r in rows], color="tab:orange")
    axes[1].set_ylabel("Best lift constraint violation\n(0 = feasible)")
    axes[1].set_xlabel("Completed VSPAERO evaluations")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Wing optimization: {status} | {len(rows)} evaluations")
    fig.savefig(folder / "convergence.png", dpi=150)
    return best


class ConvergenceMonitor(Callback):
    def __init__(self, generations, patience=0, relative_tolerance=1e-4):
        super().__init__()
        self.generations = generations
        self.patience = patience
        self.relative_tolerance = relative_tolerance
        self.reference_cd = None
        self.stale_generations = 0
        self.stopped = False

    def notify(self, algorithm):
        problem = algorithm.problem
        best = save_progress(problem.output_dir, list(problem.records.values()))
        if best:
            if self.reference_cd is None or self.reference_cd - best["CD"] > self.relative_tolerance * abs(self.reference_cd):
                self.reference_cd = best["CD"]
                self.stale_generations = 0
            else:
                self.stale_generations += 1
        feasible_count = int(np.sum(algorithm.pop.get("CV") <= 0))
        message = f"CD={best['CD']:.7f}, CL={best['CL']:.5f}" if best else "none yet"
        print(f"\n[GENERATION {algorithm.n_gen}/{self.generations}] "
              f"evaluations={len(problem.records)} | feasible population={feasible_count}/{len(algorithm.pop)} | "
              f"best feasible: {message} | stagnant generations={self.stale_generations}\n"
              f"Plot: {problem.output_dir / 'convergence.png'}", flush=True)
        if self.patience and best and self.stale_generations >= self.patience:
            self.stopped = True
            algorithm.termination.terminate()
            print("Stopping: best feasible CD has stalled. This is not proof of a global optimum.", flush=True)


class WingProblem(ElementwiseProblem):
    def __init__(self, settings, output_dir, runner=vsp_run):
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=LOWER, xu=UPPER)
        self.settings = settings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.records = {}

    def evaluate_design(self, x, *, strict=False):
        x = np.asarray(x, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)) or np.any(x < LOWER) or np.any(x > UPPER):
            raise ValueError("Design must contain three finite values within bounds")
        key = tuple(x.tolist())
        if key in self.records:
            return self.records[key]
        folder = self.output_dir / f"eval_{len(self.records):05d}"
        folder.mkdir()
        inputs = build_inputs(x, self.settings)
        write_json(folder / "inputs.json", inputs)
        print(f"\n[EVALUATION {len(self.records) + 1}] taper={x[0]:.4f}, sweep={x[1]:.3f}, alpha={x[2]:.3f}", flush=True)
        record = {"design": dict(zip(VARIABLES, key)), "directory": str(folder),
                  "status": "failed", "objective": 1e6, "constraint": 1e6}
        try:
            result = self.runner(inputs, out_dir=folder, mach=self.settings.mach,
                                 reynolds_cref=self.settings.reynolds)
            cl, cd = read_coefficients(result, x[2])
            violation = abs(cl - self.settings.target_cl) - self.settings.cl_tolerance
            record.update(status="ok", CL=cl, CD=cd, L_over_D=cl / cd,
                          objective=cd, constraint=violation,
                          feasible=violation <= 0, vsp3_path=result.get("vsp3_path"),
                          notes=result.get("notes", []))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if strict:
                write_json(folder / "result.json", record)
                raise
            print(f"Evaluation failed in {folder}: {error}", file=sys.stderr)
        self.records[key] = record
        write_json(folder / "result.json", record)
        with (self.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        feasible = [r for r in self.records.values() if r.get("feasible")]
        best = min(feasible, key=lambda r: r["CD"]) if feasible else None
        if best:
            write_json(self.output_dir / "best_design.json", best)
            write_json(self.output_dir / "best_inputs.json", build_inputs([best["design"][n] for n in VARIABLES], self.settings))
        print(f"[COMPLETED {len(self.records)}] status={record['status']} | "
              f"CL={record.get('CL')} CD={record.get('CD')} | "
              f"lift violation={max(0, record['constraint']):.6g} | "
              f"best feasible CD={best['CD'] if best else 'none yet'}", flush=True)
        return record

    def _evaluate(self, x, out, *args, **kwargs):
        record = self.evaluate_design(x)
        out["F"] = [record["objective"]]
        out["G"] = [record["constraint"]]  # pymoo feasibility: G <= 0.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openvsp-root", default=os.environ.get("OPENVSP_ROOT"))
    parser.add_argument("--output", type=Path, default=ROOT / "vsp_runs" / "shape_optimization")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="Run only one baseline VSPAERO evaluation")
    parser.add_argument("--report", type=Path, help="Rebuild convergence plot and best exports from a run folder; no solver calls")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many stagnant generations after feasibility (0 disables)")
    parser.add_argument("--relative-tolerance", type=float, default=1e-4, help="Relative CD improvement that resets patience")
    for name in ("area", "span", "target_cl", "cl_tolerance", "mach", "reynolds"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Settings(), name))
    args = parser.parse_args(argv)
    if args.report:
        records = []
        for line in (args.report / "history.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping incomplete history line", file=sys.stderr)
        best = save_progress(args.report, records, status="saved_history_report")
        print(f"Completed evaluations: {len(records)}; best feasible CD: {best['CD'] if best else 'none'}")
        print(f"Plot: {args.report / 'convergence.png'}")
        return 0
    if args.patience < 0 or not math.isfinite(args.relative_tolerance) or args.relative_tolerance < 0:
        parser.error("patience and relative-tolerance must be nonnegative and finite")
    if args.population < 4 or args.generations < 1:
        parser.error("population must be >= 4 and generations must be >= 1")
    # Convenience for this checkout; other machines should pass the path explicitly.
    local_install = ROOT.parent / "OpenVSP-3.51.3-win64"
    if not args.openvsp_root and local_install.is_dir():
        args.openvsp_root = str(local_install)
    if not args.openvsp_root:
        parser.error("Pass --openvsp-root or set OPENVSP_ROOT to your OpenVSP installation")
    try:
        settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=args.output.resolve()))
    write_json(folder / "settings.json", {**asdict(settings), "population": args.population,
               "generations": args.generations, "seed": args.seed,
               "patience": args.patience, "relative_tolerance": args.relative_tolerance,
               "variables": VARIABLES, "lower": LOWER.tolist(), "upper": UPPER.tolist(),
               "python": sys.executable})
    print(f"Results: {folder}", flush=True)
    print(f"Budget: {args.generations} generations x {args.population} candidates (up to about "
          f"{args.generations * args.population} solver calls). Each candidate produces VSPAERO output.", flush=True)
    problem = WingProblem(settings, folder)
    baseline = problem.evaluate_design(BASELINE, strict=True)
    write_json(folder / "baseline.json", baseline)
    print(f"Baseline: CL={baseline['CL']:.6f}, CD={baseline['CD']:.6f}; feasible={baseline['feasible']}")
    if args.check:
        return 0
    # Include the checked baseline and cache its result; no repeated solver call.
    rng = np.random.default_rng(args.seed)
    sampling = rng.uniform(LOWER, UPPER, (args.population, 3))
    sampling[0] = BASELINE
    monitor = ConvergenceMonitor(args.generations, args.patience, args.relative_tolerance)
    status = "generation_budget_reached"
    try:
        minimize(problem, GA(pop_size=args.population, sampling=sampling, eliminate_duplicates=True),
                 termination=("n_gen", args.generations), seed=args.seed, verbose=False, callback=monitor)
        if monitor.stopped:
            status = "stagnation_stop"
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nInterrupted; saving completed evaluations and best feasible design.", flush=True)
    except Exception:
        status = "failed"
        raise
    finally:
        save_progress(folder, list(problem.records.values()), status=status)
    if status == "interrupted":
        return 130
    feasible = [r for r in problem.records.values() if r.get("feasible", False)]
    if not feasible:
        print("No feasible design found. Inspect history.jsonl; consider more generations or revised alpha bounds.")
        return 2
    best = min(feasible, key=lambda record: record["CD"])
    write_json(folder / "best_design.json", best)
    write_json(folder / "best_inputs.json", build_inputs([best["design"][name] for name in VARIABLES], settings))
    print(f"Best feasible design: {best['design']}\nCL={best['CL']:.6f}, CD={best['CD']:.6f}")
    print(f"OpenVSP model: {best['vsp3_path']}\nSummary: {folder / 'summary.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
