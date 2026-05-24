"""
Multi-objective GA optimization (energy_trip, t_trip) using NSGA-II (pymoo)
for small eVTOL UAVs.

pip install pymoo ambiance matplotlib numpy aerosandbox
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere

# ---- Your existing imports ----
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover, power_cruise, power_climb
from DesignOS.EVTOLMDO.Model_functions.Weight_model import (
    M_wing, M_fuselage, M_battery, M_motor
)

# ---- GA library (pymoo) ----
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# =========================
# 1) PARAMETERS
# =========================
# Aircraft
t_hover = 300 / 3600        # [h]
range_cruise = 50e3         # [m]
CD0 = 0.020
CL_max = 1.5 
stall_margin = 10 # [m/s]
# Wing
n = 1                      # ultimate load factor [-]
sweep = 0
taper_ratio = 1
t_over_c = 0.12
e = 0.85
# Fuselage
l_f = 1.0                  # [m]
w_f = 0.2
d_f = 0.2
# Battery
rho_batt = 180.0  # [Wh/kg]
e_usable = 0.80
# Electric motor
eta_elec = 0.9
rho_motor = 3200 # [W/kg]
# Hover rotor
num_rotors_hover = 4  
eta_prop_hover = 0.7
eta_hover = eta_elec * eta_prop_hover
# Cruise rotor
num_rotors_cruise = 1
eta_prop_cruise = 0.7
eta_cruise = eta_elec * eta_prop_cruise
# Mission
b_cruise = 1.0
b_climb = 1.0
h_cruise = 80              # [m]
h_climb = 20               # [m]
theta_climb = math.radians(10)  # [rad] (IMPORTANT: radians)
h_hover = 20               # [m]
rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
rho_hover = float(np.squeeze(Atmosphere(h_hover).density))
g = 9.81
# Mass constants
m_payload = 1.00
m_flight_controller = 0.073
m_flight_computer = 0.080
m_avionics = 0.25
m_electronics = 0.25
m_servos = 0.012 * 7
# Constraints
MTOM_max = 2.5
b_wing_max = 2.5
V_min = 5.0
# Equality tolerance for MTOM == mass_total
tol_mass_kg = 0.01  


# =========================
# 2) NUMERIC EVALUATOR
# =========================
def evaluate_trip(x: np.ndarray):
    """
    x = [MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover]

    Returns:
      F = [energy_trip, t_trip]  (minimize both)
      G = constraints (feasible if G <= 0)
      debug dict
    """
    MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover = [float(v) for v in x]

    # ---- Geometry ----
    b_wing = math.sqrt(S * AR)
    c = S / b_wing if b_wing > 0 else 1e30
    wing_loading = (MTOM * g) / S

    # ---- Aerodynamics (for sanity prints) ----
    K = 1.0 / (math.pi * e * AR)
    CL_cruise = (2 * b_cruise * MTOM * g) / (rho_cruise * S * V_cruise**2)
    CD_cruise = CD0 + K * CL_cruise**2
    LD_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else float("inf")
    V_stall = math.sqrt((2 * MTOM * g) / (CL_max * rho_cruise * S)) 

    # ---- Power ----
    P_hover = power_hover(
        MTOM=1.15*MTOM, rho=rho_hover, R_prop=R_hover, num_rotors=num_rotors_hover, eta_h=eta_hover
    )
    P_cruise = power_cruise(
        MTOM=MTOM, b_cruise=b_cruise, AR=AR, e=e, CD0=CD0,
        rho_cruise=rho_cruise, S=S, V_cruise=V_cruise,
        R_cruise=R_cruise, eta_cruise=eta_cruise, num_rotors_cruise=num_rotors_cruise
    )
    P_climb = power_climb(
        MTOM=MTOM, b_climb=b_climb, AR=AR, e=e, CD0=CD0,
        rho_climb=rho_climb, S=S, V_climb=V_climb, theta_climb=theta_climb,
        R_cruise=R_cruise, eta_cruise=eta_cruise, num_rotors_cruise=num_rotors_cruise
    )

    # ---- Time ----
    t_cruise = (range_cruise / V_cruise) / 3600.0
    t_climb = ((h_cruise - h_climb) / (V_climb * math.sin(theta_climb))) / 3600.0
    t_trip = t_cruise + t_climb + t_hover

    # ---- Energy (Wh) ----
    energy_hover = P_hover * t_hover
    energy_cruise = P_cruise * t_cruise
    energy_climb = P_climb * t_climb
    energy_trip = energy_hover + energy_climb + energy_cruise

    # ---- Mass ----
    mass_battery = M_battery(E_trip=energy_trip, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
    mass_motor_cruise = M_motor(P_out=P_climb, rho_motor=rho_motor)
    mass_motor_hover = M_motor(P_out=P_hover, rho_motor=rho_motor)
    mass_motor = mass_motor_cruise + mass_motor_hover
    mass_wing = M_wing(
        b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio,
        t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise
    )
    mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)

    mass_total = (
        mass_motor + mass_wing + m_payload + mass_fuselage + mass_battery +
        m_avionics + m_electronics + m_flight_controller + m_flight_computer
    )

    # ---- Constraints (G <= 0 feasible) ----
    G = np.array([
        MTOM - MTOM_max,                           # MTOM <= MTOM_max
        V_min - V_cruise,                          # V_cruise >= V_min
        V_min - V_climb,                           # V_climb >= V_min
        stall_margin - (V_cruise - V_stall),       # V_cruise - V_stall >= stall_margin
        b_wing - b_wing_max,                       # b_wing <= b_wing_max
        abs(MTOM - mass_total) - tol_mass_kg       # MTOM == mass_total (toleranced)
    ], dtype=float)

    F = np.array([energy_trip, t_trip], dtype=float)

    # Guard against numerical issues
    if (not np.all(np.isfinite(F))) or (not np.all(np.isfinite(G))):
        F = np.array([1e30, 1e30], dtype=float)
        G = np.array([1e30] * len(G), dtype=float)

    debug = {
        "MTOM": MTOM,
        "mass_total": mass_total,
        "mass_total_minus_MTOM": mass_total - MTOM,
        "S": S,
        "AR": AR,
        "b_wing": b_wing,
        "c": c,
        "wing_loading": wing_loading,
        "V_cruise": V_cruise,
        "V_climb": V_climb,
        "V_stall": V_stall,
        "R_cruise": R_cruise,
        "R_hover": R_hover,
        "CL_cruise": CL_cruise,
        "CD_cruise": CD_cruise,
        "LD_cruise": LD_cruise,
        "P_hover": P_hover,
        "P_cruise": P_cruise,
        "P_climb": P_climb,
        "t_cruise": t_cruise,
        "t_climb": t_climb,
        "t_trip": t_trip,
        "energy_trip": energy_trip,
        "mass_battery": mass_battery,
        "mass_fuselage": mass_fuselage,
        "mass_wing": mass_wing,
        "hover_motor_mass": mass_motor_hover / num_rotors_hover, 
        "cruise_motor_mass": mass_motor_cruise / num_rotors_cruise
    }

    return F, G, debug


# =========================
# 3) PYMOO PROBLEM WRAPPER
# =========================
class EVTOLUAVTripProblem(Problem):
    def __init__(self):
        # Decision vector: [MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover]
        xl = np.array([
            1.0,     # MTOM [kg]
            0.1,     # S [m^2]
            1.0,     # V_cruise [m/s]
            1.0,     # V_climb [m/s]
            1.0,     # AR [-]
            1*0.0254,# R_cruise [m]
            1*0.0254 # R_hover [m]
        ], dtype=float)

        xu = np.array([
            5.0,    # MTOM [kg]
            10.0,    # S [m^2]
            25.0,    # V_cruise [m/s]
            25.0,    # V_climb [m/s]
            8.0,     # AR [-]
            4*0.0254,# R_cruise [m]
            4*0.0254 # R_hover [m]
        ], dtype=float)

        super().__init__(n_var=7, n_obj=2, n_ieq_constr=6, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2), dtype=float)
        G = np.zeros((X.shape[0], 6), dtype=float)
        for i in range(X.shape[0]):
            f, g, _ = evaluate_trip(X[i, :])
            F[i, :] = f
            G[i, :] = g
        out["F"] = F
        out["G"] = G


# =========================
# 4) RUN NSGA-II
# =========================
def main():
    problem = EVTOLUAVTripProblem()

    algorithm = NSGA2(
        pop_size=200,
        eliminate_duplicates=True
    )

    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", 200),
        seed=1,
        verbose=True
    )

    X_pareto = res.X
    F_pareto = res.F  # [:,0] energy_trip, [:,1] t_trip

    if X_pareto is None or F_pareto is None:
        print("\nNo feasible Pareto set found (res.X is None).")
        return

    print("\n--- Sanity checks on a few Pareto solutions ---")
    n_check = min(5, len(X_pareto))
    for i in range(n_check):
        x = X_pareto[i]
        f, g, dbg = evaluate_trip(x)

        print(f"\nPareto[{i:02d}] x = {x}")
        print(f"  Objectives: energy_trip={f[0]:.3f} Wh, t_trip={f[1]*60:.3f} min")
        print(f"  mass_total - MTOM: {dbg['mass_total_minus_MTOM']:+.4f} kg")
        print(f"  b_wing: {dbg['b_wing']:.3f} m (<= {b_wing_max})")
        print(f"  CL_cruise: {dbg['CL_cruise']:.3f}, CD_cruise: {dbg['CD_cruise']:.4f}, L/D: {dbg['LD_cruise']:.2f}")
        print(f"  P_hover: {dbg['P_hover']:.1f} W | P_cruise: {dbg['P_cruise']:.1f} W | P_climb: {dbg['P_climb']:.1f} W")
        print(f"  Battery mass: {dbg['mass_battery']*1000:.1f} g")
        print(f"  Max constraint violation: {np.max(np.maximum(0.0, g)):.6f}")

    print("\n====================")
    print("Pareto set size:", len(X_pareto))
    print("====================\n")

    # Plot Pareto front
    plt.figure()
    plt.scatter(F_pareto[:, 0], F_pareto[:, 1] * 60.0, s=18)  # minutes
    plt.xlabel("energy_trip [Wh]")
    plt.ylabel("t_trip [min]")
    plt.title("Pareto Front: energy vs time (NSGA-II)")
    plt.grid(True)
    plt.show()

    # Optional: choose a compromise point (normalized weighted sum)
    wE, wT = 0.5, 0.5
    E_norm = F_pareto[:, 0] / np.min(F_pareto[:, 0])
    T_norm = F_pareto[:, 1] / np.min(F_pareto[:, 1])
    score = wE * E_norm + wT * T_norm
    best_idx = int(np.argmin(score))

    x_best = X_pareto[best_idx]
    f_best = F_pareto[best_idx]
    _, g_best, dbg_best = evaluate_trip(x_best)

    print("\n--- Chosen compromise (weighted normalized sum) ---")
    print("best_idx:", best_idx)
    print("x_best [MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover] =", x_best)
    print("f_best [energy_trip (Wh), t_trip (h)] =", f_best)
    print(f"Max constraint violation: {np.max(np.maximum(0.0, g_best)):.6f}")

    # Nice summary
    print("\n### --- Best Design Summary --- ###")
    print(f"MTOM: {dbg_best['MTOM']:.3f} kg")
    print(f"S: {dbg_best['S']:.3f} m^2 | AR: {dbg_best['AR']:.3f} | Wing loading: {(dbg_best['MTOM']*9.81) / dbg_best['S']} N/m^2")
    print(f"b_wing: {dbg_best['b_wing']:.3f} m | c: {dbg_best['c']:.3f} m")
    print(f"V_cruise: {dbg_best['V_cruise']:.3f} m/s | V_climb: {dbg_best['V_climb']:.3f} m/s | V_stall: {dbg_best['V_stall']:.3f} m/s")
    print(f"R_cruise: {dbg_best['R_cruise']:.4f} m | R_hover: {dbg_best['R_hover']:.4f} m")
    print(f"t_trip: {dbg_best['t_trip']*60:.2f} min | t_cruise: {dbg_best['t_cruise']*60:.2f} min | t_climb: {dbg_best['t_climb']*60:.2f} min")
    print(f"energy_trip: {dbg_best['energy_trip']:.2f} Wh")
    print(f"mass_total - MTOM: {dbg_best['mass_total_minus_MTOM']:+.6f} kg")
    print(f"CL_cruise: {dbg_best['CL_cruise']:.3f} | CD_cruise: {dbg_best['CD_cruise']:.4f} | L/D: {dbg_best['LD_cruise']:.2f}")
    print(f"Hover power: {dbg_best['P_hover']:.3f} W | Cruise power: {dbg_best['P_cruise']:.4f} W | Climb power: {dbg_best['P_climb']:.2f} W")
    print(f"Hover motor power: {dbg_best['P_hover'] / num_rotors_hover} W")
    print(f"Cruise motor power: {dbg_best['P_climb'] / num_rotors_cruise} W")
    print(f"Battery mass: {dbg_best['mass_battery']} kg")
    print(f"Hover motor mass: {dbg_best['hover_motor_mass']} kg")
    print(f"Cruise motor mass: {dbg_best['cruise_motor_mass']} kg")
    print(f"Fuselage mass: {dbg_best['mass_fuselage']} kg")
    print(f"Wing mass: {dbg_best['mass_wing']} kg")
    print(f"Hover disk loading: {(dbg_best['MTOM']*9.81) / (num_rotors_hover * math.pi * dbg_best['R_hover']**2)} N/m^2")


if __name__ == "__main__":
    main()
