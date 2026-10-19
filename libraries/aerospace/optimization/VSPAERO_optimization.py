"""Fixed-span, fixed-area wing optimization using an injected VSPAERO runner."""

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from libraries.aerospace.geometry.aircraft_definition import Aircraft, Wing, WingXsec, Airfoil


def _scalar(value, name):
    values = np.asarray(value, dtype=float)
    if values.size != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain exactly one finite value")
    return float(values.reshape(-1)[0])


class VSPAeroOptimProblem(ElementwiseProblem):
    """Seven variables: alpha, croot, t1, c1, t2, c2, t3.

    Root twist is zero; t1/t2/t3 are absolute outboard-station twists in degrees.
    The runner must interpret each WingXsec.twist_deg using that convention.
    Sections describe one semispan, mirrored by the runner. Tip chord is derived
    from the full-wing trapezoidal planform area, before applying twist.
    """

    def __init__(self, V, h, b_set, S_set, runner, *, CL_target=1.0,
                 CL_tolerance=1e-3, station_fractions=(0, 1/3, 2/3, 1),
                 chord_bounds=(0.05, 2.0), min_chord_drop=1e-3):
        self.V, self.h = _scalar(V, "V"), _scalar(h, "h")
        self.span, self.area = _scalar(b_set, "b_set"), _scalar(S_set, "S_set")
        self.CL_target = _scalar(CL_target, "CL_target")
        self.CL_tolerance = _scalar(CL_tolerance, "CL_tolerance")
        self.min_chord_drop = _scalar(min_chord_drop, "min_chord_drop")
        self.cmin, self.cmax = map(float, chord_bounds)
        stations = np.asarray(station_fractions, dtype=float)
        if self.V <= 0 or self.span <= 0 or self.area <= 0:
            raise ValueError("V, b_set and S_set must be positive")
        if self.CL_tolerance <= 0 or self.min_chord_drop <= 0:
            raise ValueError("CL_tolerance and min_chord_drop must be positive")
        if not (np.isfinite(self.cmin) and np.isfinite(self.cmax)
                and 0 < self.cmin < self.cmax):
            raise ValueError("chord_bounds must be finite, positive and increasing")
        if (stations.shape != (4,) or not np.all(np.isfinite(stations))
                or stations[0] != 0 or stations[-1] != 1
                or np.any(np.diff(stations) <= 0)):
            raise ValueError("station_fractions must be four increasing values from 0 to 1")
        self.widths = np.diff(stations) * self.span / 2
        # Smallest/largest feasible tapered chord distributions.
        low = self.cmin + self.min_chord_drop * np.arange(3, -1, -1)
        high = self.cmax - self.min_chord_drop * np.arange(4)
        area_low = np.dot(self.widths, low[:-1] + low[1:])
        area_high = np.dot(self.widths, high[:-1] + high[1:])
        if low[0] > self.cmax or not area_low <= self.area <= area_high:
            raise ValueError("Requested area is infeasible with these chord bounds and taper margin")
        if not callable(runner):
            raise TypeError("runner must be callable")
        self.runner = runner
        self.failed_evaluations = 0
        self.last_error = None
        super().__init__(
            n_var=7, n_obj=1, n_ieq_constr=6,
            xl=[0, self.cmin, -5, self.cmin, -5, self.cmin, -5],
            xu=[10, self.cmax, 5, self.cmax, 5, self.cmax, 5],
        )

    def chords(self, x):
        _, croot, _, c1, _, c2, _ = x
        d0, d1, d2 = self.widths
        c3 = (self.area - d0 * (croot + c1) - d1 * (c1 + c2)) / d2 - c2
        return np.array([croot, c1, c2, c3])

    def build_aircraft(self, x):
        """Use the same geometry for candidate evaluations and the final model."""
        chords = self.chords(x)
        wing = Wing(
            name="Main Wing",
            wing_sections=[
                WingXsec(span=float(width), root_chord=float(chords[i]),
                         tip_chord=float(chords[i + 1]), twist_deg=float(x[2 + 2*i]),
                         twist_loc=0, sweep_deg=0, dihedral_deg=0)
                for i, width in enumerate(self.widths)
            ],
            airfoils=[Airfoil(thick_chord=0.12, camber=0.04, camber_loc=0.4)
                      for _ in range(4)],
            x_loc=0, y_loc=0, z_loc=0, x_rot=0, y_rot=0, z_rot=0,
        )
        return Aircraft(name="Optimized Wing", wing_list=[wing])

    def _evaluate(self, x, out, *args, **kwargs):
        chords = self.chords(x)
        # pymoo requires G <= 0. A positive margin implements strict taper.
        geometry_g = np.r_[
            np.diff(chords) + self.min_chord_drop,
            self.cmin - chords[-1], chords[-1] - self.cmax,
        ] / self.cmax
        out["F"] = [np.inf]
        out["G"] = np.r_[geometry_g, 1.0]
        if np.any(geometry_g > 0):
            return  # Skip invalid or reverse-taper geometry before VSPAERO.
        aircraft = self.build_aircraft(x)
        try:
            outputs = self.runner(aircraft, self.V, alpha=float(x[0]), h=self.h)
            cl = _scalar(outputs["sweep_outputs"]["CL"], "CL")
            cd = _scalar(outputs["sweep_outputs"]["CD"], "CD")
            if cd < 0:
                raise ValueError("CD must be nonnegative")
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            self.failed_evaluations += 1
            self.last_error = str(error)
            return
        out["F"] = [cd]
        out["G"] = np.r_[geometry_g, abs(cl - self.CL_target) - self.CL_tolerance]


def wing_aero_shape_optimization(V, h, b_set, S_set, *, runner=None,
                                 CL_target=1.0, CL_tolerance=1e-3,
                                 station_fractions=(0, 1/3, 2/3, 1),
                                 chord_bounds=(0.05, 2.0), min_chord_drop=1e-3,
                                 pop_size=200, n_gen=200, seed=1, verbose=True,
                                 final_output_dir="vsp_runs/optimized_wing"):
    """Minimize CD at CL_target within CL_tolerance.

    Defaults to tests.test_vsp_run.vsp_run. A custom runner can be passed with
    signature (aircraft, V, alpha, *, h=0). It must
    return sweep_outputs containing scalar (or length-one) CL and CD values.
    It must use the actual full-wing area/span as aerodynamic references.

    Returns [alpha, croot, t1, c1, t2, c2, t3, c3], or a failure dictionary.
    Unlike the original sketch, the three twists are outboard-station values;
    root twist is fixed at zero. All chord/length inputs use consistent units.

    After success, reruns the selected design. The default VSPAERO runner saves
    final_output_dir/optimized_wing.vsp3 for opening in the OpenVSP GUI (replaced
    on subsequent runs). Custom runners are rerun with their usual signature
    and retain control of their own output paths. The returned vector is unchanged.
    """
    if runner is None:
        from functools import partial
        from tests.test_vsp_run import vsp_run
        runner = vsp_run
        final_runner = partial(vsp_run, out_dir=final_output_dir,
                               vsp3_name="optimized_wing.vsp3")
    else:
        final_runner = runner
    problem = VSPAeroOptimProblem(
        V, h, b_set, S_set, runner, CL_target=CL_target,
        CL_tolerance=CL_tolerance, station_fractions=station_fractions,
        chord_bounds=chord_bounds, min_chord_drop=min_chord_drop,
    )
    res = minimize(problem, GA(pop_size=pop_size, eliminate_duplicates=True),
                   termination=("n_gen", n_gen), seed=seed, verbose=verbose)
    if res.X is None or res.F is None or not np.all(np.isfinite(res.F)):
        return {
            "status": "no_solution_found",
            "message": "No feasible solution found; inspect bounds, CL target and solver failures.",
            "failed_evaluations": problem.failed_evaluations,
            "last_error": problem.last_error,
        }
    x_best = np.asarray(res.X, dtype=float)

    # Rebuild and run the optimum, so OpenVSP's last model is the selected
    # design rather than whichever candidate happened to be evaluated last.
    optimal_aircraft = problem.build_aircraft(x_best)
    final_outputs = final_runner(optimal_aircraft, V, alpha=float(x_best[0]), h=h)
    model_path = final_outputs.get("vsp3_path")
    if model_path:
        print(f"Optimal wing model: {model_path}")
        print("Open this .vsp3 file in OpenVSP to visualize the optimized wing.")
    return np.r_[x_best, problem.chords(x_best)[-1]]
