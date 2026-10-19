"""Opt-in real solver check: python -m tests.run_vspaero_optimization_smoke.

Runs a small GA, not a convergence study. Saves inputs/results for each actual
solver call and a summary under vsp_runs/optimization_smoke/<unique directory>.
"""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from libraries.aerospace.optimization.VSPAERO_optimization import (
    VSPAeroOptimProblem,
)
from tests.test_vsp_run import vsp_run


def main():
    parent = Path("vsp_runs/optimization_smoke").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    records = []

    def runner(aircraft, V, alpha, *, h=0):
        evaluation = folder / f"eval_{len(records):04d}"
        result = vsp_run(aircraft, V, alpha, h=h, out_dir=evaluation)
        record = {"aircraft": asdict(aircraft), "result": result}
        (evaluation / "result.json").write_text(json.dumps(record, indent=2))
        records.append(record)
        return result

    # A feasible seed from the initial real-solver exploratory run. Seeding
    # keeps this integration check short and repeatable, not a global search.
    initial = np.array([7.106948794398168, 1.2646385548439192, 1.5042812027625447,
                        1.2132897343401527, 4.176922571709127,
                        0.8115165970801177, 0.06913353247997733])
    sampling = np.tile(initial, (6, 1))
    sampling[:, 0] += np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    problem = VSPAeroOptimProblem(30, 100, 6, 6, runner,
                                 CL_target=1, CL_tolerance=0.01)
    result = minimize(
        problem, GA(pop_size=6, sampling=sampling, eliminate_duplicates=True),
        termination=("n_gen", 3), seed=1, verbose=True,
    )
    summary = {"settings": {"V": 30, "h": 100, "span": 6, "area": 6,
                            "CL_target": 1, "CL_tolerance": 0.01,
                            "pop_size": 6, "n_gen": 3, "seed": 1,
                            "initial_design": initial.tolist()},
               "solver_calls": len(records),
               "failed_evaluations": problem.failed_evaluations}
    if result.X is None:
        summary.update(status="no_solution_found", last_error=problem.last_error)
    else:
        x_best = np.r_[result.X, problem.chords(result.X)[-1]]
        summary["x_best"] = x_best.tolist()
        feasible = [r for r in records
                    if abs(r["result"]["sweep_outputs"]["CL"][0] - 1) <= 0.01]
        best = min(feasible, key=lambda r: r["result"]["sweep_outputs"]["CD"][0])
        summary["best_evaluation"] = best
        summary["status"] = "passed"
        assert np.all(result.G <= 0)
        assert problem.failed_evaluations == 0, problem.last_error
        assert np.isclose(result.F[0], best["result"]["sweep_outputs"]["CD"][0])
        chords = x_best[[1, 3, 5, 7]]
        assert np.all(np.diff(chords) <= -0.001)
        assert np.isclose(sum(chords[:-1] + chords[1:]), 6)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", folder / "summary.json", flush=True)
    assert summary["status"] == "passed", summary


if __name__ == "__main__":
    main()
