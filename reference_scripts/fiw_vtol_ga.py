"""
Multi-objective (energy_trip, time_trip) genetic algorithm optimization (NSGA-II) using pymoo.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere

# ---- Model imports ----
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover
from DesignOS.EVTOLMDO.Model_functions.Thrust_model import thrust_required_cruise, thrust_required_climb
from DesignOS.EVTOLMDO.Model_functions.Cost_model import economic_analysis
from DesignOS.EVTOLMDO.Model_functions.Weight_model import (
    M_rotor, M_motor, M_systems, M_payload, M_wing, M_fuselage, M_furnish, M_engine, M_fuel_jet_cruise, M_battery
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
range_cruise = 700e3       # [m]
CD0 = 0.020
t_hover = 120 / 3600       # [h]
# Wing
n = 1                      # ultimate load factor [-]
sweep = 0
taper_ratio = 1
t_over_c = 0.12
e = 0.85
# Fuselage
l_f = 6.0                  # [m]
w_f = 1.0
d_f = 3.0
# Battery
rho_batt = 180.0           # [Wh/kg]
c_rate = 10
e_usable = 0.80
# Electric motor
eta_elec = 0.9
rho_motor = 3225.806       # [W/kg]
# Rotor
num_rotors = 4
k_rotor = 1.5
eta_prop_hover = 0.7
eta_hover = eta_elec * eta_prop_hover
# Fuel System
rho_fuel_sys = 0.95
rho_fuel = 12000           # [Wh/kg]
rho_fuel_vol = 807.5 / 1000 # [kg/L]
# Turbofan Engine
TSFC = 1.4e-5              # [kg/(N*s)]
T_to_W = 4
# Payload
m_pax = 82
n_seats = 4
m_lugg = 16
n_crew = 0
# Mission
b_cruise = 0.95
b_climb = 1.0
h_cruise = 1000            # [m]
h_climb = 100              # [m]
theta_climb = 10           # [deg]
h_hover = 100              # [m]
rho_sl = float(np.squeeze(Atmosphere(0).density))
g = 9.81
# Economic
n_wd = 260 # Number of operating days per year of one eVTOL [days]
T_D = 8 # Daily operation window of one eVTOL [-]
C_rate_charge = 2 
P_energy = 0.11 # Electricity price [$/kWh]
j = 0.0796 # Annuity factor [-]
P_empty = 1664.1 # Price of eVTOL per empty weight [$/kg]
x_ins = 0.06 # Insurance factor
x_irs = 0.14 # Reservation and sales factor
x_iap = 0.02 # Advertising and publicity factor
x_iga = 0.06 # General and administration factor
MF = 0.6 # Maintenance man-hours per flight hour ratio
MWR = 63.71 # Maintenance wrap-rate [$/hr]
P_batt = 150.6 # Energy specific battery acquisition cost [$/kWh]
N_cycles = 420 # Number of life cycles available per battery
n_ac = 1 # number of aircraft controlled by one pilot
P_pilot = 45.2 # hourly salary of a pilot [$/hr]
fare = 3.0 # fare per km [$/km]
# Equality constraint tolerance (MTOM == mass_total)
tol_mass_kg = 5.0


# =========================
# 2) NUMERIC EVALUATOR
# =========================
def evaluate_trip(x: np.ndarray):
    """
    x = [MTOM, S, V_cruise, V_climb, AR, A_disk]
    Returns:
      F = [energy_trip, time_trip] (both minimized)
      G = constraints array where feasible means G <= 0
      debug dict (sanity prints)
    """
    MTOM, S, V_cruise, V_climb, AR, A_disk = x

    # Atmosphere
    rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
    rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
    rho_hover = float(np.squeeze(Atmosphere(h_hover).density))

    # Geometry
    R_rotor = math.sqrt((A_disk / num_rotors) / math.pi)
    b_wing = math.sqrt(S * AR)
    c = S / b_wing

    # Aerodynamics (same as M_fuel_jet_cruise)
    CL_cruise = (2 * b_cruise * MTOM * g) / (rho_cruise * S * V_cruise**2)
    K = 1.0 / (math.pi * e * AR)
    CD_cruise = CD0 + K * CL_cruise**2
    LD_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else float("inf")

    # Fuel exponent argument exactly as in M_fuel_jet_cruise.py
    # ff = exp( R / ( (V/(TSFC*g)) * (CL/CD) ) )
    fuel_exp_arg = float("nan")
    try:
        denom = (V_cruise / (TSFC * g)) * (CL_cruise / CD_cruise)
        fuel_exp_arg = range_cruise / denom
    except Exception:
        fuel_exp_arg = float("nan")

    # Guard: if exponent is too large, exp() will overflow -> penalize and move on
    # (math.exp overflows around arg ~ 709 for double precision)
    if (not np.isfinite(fuel_exp_arg)) or (fuel_exp_arg > 700):
        F = np.array([1e30, 1e30], dtype=float)
        G = np.array([1e30] * 6, dtype=float)
        debug = {
            "mass_total": float("nan"),
            "MTOM": MTOM,
            "mass_total_minus_MTOM": float("nan"),
            "b_wing": b_wing,
            "A_disk_over_S": A_disk / S,
            "CL_cruise": CL_cruise,
            "CD_cruise": CD_cruise,
            "LD_cruise": LD_cruise,
            "mass_fuel_cruise": float("nan"),
            "fuel_exp_arg": fuel_exp_arg,
        }
        return F, G, debug

    # Thrust
    T_SL_cruise = thrust_required_cruise(
        k=1, rho_alt=rho_cruise, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e,
        n=n, b=b_cruise, MTOM=MTOM, S=S, V=V_cruise
    )
    T_SL_climb = thrust_required_climb(
        k=1, rho_alt=rho_climb, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e,
        b=b_climb, MTOM=MTOM, S=S, V=V_climb, theta=theta_climb
    )
    T_SL = max(T_SL_cruise, T_SL_climb)

    # Power (hover)
    P_hover = 1.15 * power_hover(MTOM=MTOM, rho=rho_hover, R_prop=R_rotor, num_rotors=num_rotors, eta_h=eta_hover)

    # Fuel mass (calls your function)
    mass_fuel_cruise = M_fuel_jet_cruise(
        TSFC=TSFC, rho_cruise=rho_cruise, MTOM=MTOM, S=S, V=V_cruise, AR=AR, e=e,
        CD0=CD0, b=b_cruise, R=range_cruise, g=g
    )
    mass_fuel_climb = MTOM * 0.05
    mass_fuel = mass_fuel_cruise + mass_fuel_climb

    # Energy
    energy_battery = max(P_hover * t_hover, P_hover / c_rate)
    energy_fuel = mass_fuel * rho_fuel
    energy_trip = energy_battery + energy_fuel

    # Time
    time_cruise = (range_cruise / V_cruise) / 3600.0
    time_climb = ((h_cruise - h_climb) / (V_climb * math.sin(math.radians(theta_climb)))) / 3600.0
    time_trip = time_cruise + time_climb

    # Mass breakdown
    mass_motor = M_motor(P_out=P_hover, rho_motor=rho_motor)
    mass_rotor = M_rotor(num_cruise=0, num_hover=num_rotors, R_rotor_cruise=R_rotor, R_rotor_hover=R_rotor, k_rotor=k_rotor)
    mass_systems = M_systems(n=n, MTOM=MTOM, l_f=l_f, b=b_wing)
    mass_wing = M_wing(b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio, t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise)
    mass_payload = M_payload(M_pax=m_pax, M_lug=m_lugg, n_seats=n_seats)
    mass_engine = M_engine(T_to_W=T_to_W, T_cruise=T_SL)
    mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
    mass_furnish = M_furnish(MTOM=MTOM, n_crew=n_crew, V_cruise=V_cruise, b=b_wing, c=c)
    mass_fuel_sys = mass_fuel / rho_fuel_sys
    mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)

    mass_total = (
        mass_motor + mass_rotor + mass_systems + mass_wing + mass_payload + mass_engine +
        mass_fuselage + mass_furnish + mass_fuel_sys + mass_battery
    )

    # Economic analysis
    # Approximate empty mass by removing payload, fuel system, and battery masses from total
    mass_empty_cost = max(mass_total - (mass_payload + mass_fuel_sys + mass_battery), 0.0)
    TOC_trip, profit_trip, profit_annual = economic_analysis(
        n_wd=n_wd,
        T_D=T_D,
        C_rate_charge=C_rate_charge,
        E_trip=energy_trip / 1000,            # kWh
        E_battery=energy_battery / 1000,      # kWh
        t_trip=time_trip,                     # hours
        P_energy=P_energy,
        j=j,
        P_empty=P_empty,
        M_empty=mass_empty_cost,
        x_ins=x_ins,
        x_irs=x_irs,
        x_iap=x_iap,
        x_iga=x_iga,
        MF=MF,
        MWR=MWR,
        P_batt=P_batt,
        rho_batt=rho_batt,
        M_batt=mass_battery,
        N_cycles=N_cycles,
        D_trip=range_cruise / 1000,           # km
        n_ac=n_ac,
        P_pilot=P_pilot,
        fare=fare
    )


    # Constraints formatted as G <= 0
    G = np.array([
        MTOM - 5700.0,                       # MTOM <= 5700
        120.0 - V_cruise,                    # V_cruise > 120
        100.0 - V_climb,                     # V_climb > 100
        b_wing - 15.0,                       # b_wing <= 15
        (A_disk / S) - 0.45,                 # A_disk/S <= 0.45
        abs(MTOM - mass_total) - tol_mass_kg # MTOM == mass_total (toleranced)
    ], dtype=float)

    F = np.array([energy_trip, time_trip], dtype=float)

    # Guard against numerical issues during GA exploration
    if (not np.all(np.isfinite(F))) or (not np.all(np.isfinite(G))):
        F = np.array([1e30, 1e30], dtype=float)
        G = np.array([1e30] * len(G), dtype=float)

    debug = {
        "mass_total": mass_total,
        "MTOM": MTOM,
        "mass_total_minus_MTOM": mass_total - MTOM,
        "b_wing": b_wing,
        "A_disk_over_S": A_disk / S,
        "CL_cruise": CL_cruise,
        "CD_cruise": CD_cruise,
        "LD_cruise": LD_cruise,
        "mass_fuel_cruise": mass_fuel_cruise,
        "fuel_exp_arg": fuel_exp_arg,
    }

    return F, G, debug


# =========================
# 3) PYMOO PROBLEM WRAPPER
# =========================
class EVTOLTripProblem(Problem):
    def __init__(self):
        # Decision vector: [MTOM, S, V_cruise, V_climb, AR, A_disk]
        xl = np.array([1.0,    1.0,   30.0,   30.0,  1.0,   1.0], dtype=float)
        xu = np.array([5700.0, 100.0, 300.0, 300.0, 5.0, 100.0], dtype=float)
        super().__init__(n_var=6, n_obj=2, n_ieq_constr=6, xl=xl, xu=xu)

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
    problem = EVTOLTripProblem()

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
    F_pareto = res.F  # [:,0] energy_trip, [:,1] time_trip

    if X_pareto is None or F_pareto is None:
        print("\nNo feasible Pareto set found (res.X is None).")
        return

    print("\n--- Sanity checks on a few Pareto solutions ---")
    n_check = min(5, len(X_pareto))
    for i in range(n_check):
        x = X_pareto[i]
        f, g_constr, dbg = evaluate_trip(x)

        print(f"\nPareto[{i:02d}] x = {x}")
        print(f"  Objectives: energy_trip={f[0]:.3e}, time_trip={f[1]:.4f} h")
        print(f"  mass_total - MTOM: {dbg['mass_total_minus_MTOM']:+.3f} kg")
        print(f"  b_wing: {dbg['b_wing']:.3f} m")
        print(f"  A_disk/S: {dbg['A_disk_over_S']:.4f}")
        print(f"  CL_cruise: {dbg['CL_cruise']:.4f}")
        print(f"  CD_cruise: {dbg['CD_cruise']:.5f}")
        print(f"  L/D cruise: {dbg['LD_cruise']:.3f}")
        print(f"  mass_fuel_cruise: {dbg['mass_fuel_cruise']:.3f} kg")
        fea = dbg.get("fuel_exp_arg", float("nan"))
        print(f"  fuel exp arg: {fea:.6f}" if np.isfinite(fea) else f"  fuel exp arg: {fea}")
        print(f"  Max constraint violation: {np.max(np.maximum(0.0, g_constr)):.6f}")

    print("\n====================")
    print("Pareto set size:", len(X_pareto))
    print("====================\n")

    # Show a few Pareto solutions
    for i in range(min(10, len(X_pareto))):
        MTOM, S, Vc, Vcl, ARv, Ad = X_pareto[i]
        E, T = F_pareto[i]
        print(f"[{i:02d}] MTOM={MTOM:8.2f}  S={S:7.2f}  Vc={Vc:7.2f}  Vcl={Vcl:7.2f}  "
              f"AR={ARv:5.2f}  Ad={Ad:7.2f}  |  E={E:.3e}, T={T:.4f} h")

    # Plot Pareto front
    plt.figure()
    plt.scatter(F_pareto[:, 0], F_pareto[:, 1], s=18)
    plt.xlabel("energy_trip")
    plt.ylabel("time_trip [h]")
    plt.title("Pareto Front: energy vs time (NSGA-II)")
    plt.grid(True)
    plt.show()

    # Optional: pick a single "best" compromise by normalized weighted sum
    wE, wT = 0.5, 0.5
    E_norm = F_pareto[:, 0] / np.min(F_pareto[:, 0])
    T_norm = F_pareto[:, 1] / np.min(F_pareto[:, 1])
    score = wE * E_norm + wT * T_norm
    best_idx = int(np.argmin(score))
    x_best = X_pareto[best_idx]
    f_best = F_pareto[best_idx]

    print("\n--- Chosen compromise (weighted normalized sum) ---")
    print("best_idx:", best_idx)
    print("x_best [MTOM, S, V_cruise, V_climb, AR, A_disk] =", x_best)
    print("f_best [energy_trip, time_trip] =", f_best)

    # Display detailed properties for the chosen design point
    MTOM_best, S_best, V_cruise_best, V_climb_best, AR_best, A_disk_best = [float(v) for v in x_best]

    rho_cruise_best = float(np.squeeze(Atmosphere(h_cruise).density))
    rho_climb_best  = float(np.squeeze(Atmosphere(h_climb).density))
    rho_hover_best  = float(np.squeeze(Atmosphere(h_hover).density))

    R_rotor_best = math.sqrt((A_disk_best / num_rotors) / math.pi)
    b_wing_best = math.sqrt(S_best * AR_best)
    c_best = S_best / b_wing_best
    wing_loading_best = (MTOM_best * g) / S_best

    K_best = 1.0 / (math.pi * e * AR_best)

    CL_cruise_best = (2 * b_cruise * MTOM_best * g) / (rho_cruise_best * S_best * V_cruise_best**2)
    CL_climb_best  = (2 * b_climb  * MTOM_best * g) / (rho_climb_best  * S_best * V_climb_best**2)

    CD_cruise_best = CD0 + K_best * CL_cruise_best**2
    CD_climb_best  = CD0 + K_best * CL_climb_best**2

    CL_over_CD_cruise_best = CL_cruise_best / CD_cruise_best
    CL_over_CD_climb_best  = CL_climb_best  / CD_climb_best

    T_SL_cruise_best = thrust_required_cruise(
        k=1,
        rho_alt=rho_cruise_best,
        rho_sl=float(rho_sl),
        CD0=float(CD0),
        AR=float(AR_best),
        e=float(e),
        n=float(n),
        b=float(b_cruise),
        MTOM=float(MTOM_best),
        S=float(S_best),
        V=float(V_cruise_best)
    )

    T_SL_climb_best = thrust_required_climb(
        k=1,
        rho_alt=rho_climb_best,
        rho_sl=float(rho_sl),
        CD0=float(CD0),
        AR=float(AR_best),
        e=float(e),
        b=float(b_climb),
        MTOM=float(MTOM_best),
        S=float(S_best),
        V=float(V_climb_best),
        theta=float(theta_climb)
    )

    T_SL_best = max(T_SL_cruise_best, T_SL_climb_best)

    P_hover_best = 1.15 * power_hover(
        MTOM=float(MTOM_best),
        rho=float(rho_hover_best),
        R_prop=float(R_rotor_best),
        num_rotors=int(num_rotors),
        eta_h=float(eta_hover)
    )

    # IMPORTANT: force scalar floats into the fuel function to avoid math.exp(array) errors
    mass_fuel_cruise_best = M_fuel_jet_cruise(
        TSFC=float(TSFC),
        rho_cruise=float(rho_cruise_best),
        MTOM=float(MTOM_best),
        S=float(S_best),
        V=float(V_cruise_best),
        AR=float(AR_best),
        e=float(e),
        CD0=float(CD0),
        b=float(b_cruise),
        R=float(range_cruise),
        g=float(g)
    )

    mass_fuel_climb_best = float(MTOM_best) * 0.05
    mass_fuel_best = mass_fuel_cruise_best + mass_fuel_climb_best

    energy_battery_best = max(P_hover_best * t_hover, P_hover_best / c_rate)
    energy_fuel_best = mass_fuel_best * rho_fuel
    energy_trip_best = energy_battery_best + energy_fuel_best

    time_cruise_best = (range_cruise / V_cruise_best) / 3600.0
    time_climb_best = ((h_cruise - h_climb) / (V_climb_best * math.sin(math.radians(theta_climb)))) / 3600.0
    time_trip_best = time_cruise_best + time_climb_best

    # Mass breakdown (recompute at best point)
    mass_motor_best = M_motor(P_out=P_hover_best, rho_motor=rho_motor)
    mass_rotor_best = M_rotor(num_cruise=0, num_hover=num_rotors,
                              R_rotor_cruise=R_rotor_best, R_rotor_hover=R_rotor_best, k_rotor=k_rotor)
    mass_systems_best = M_systems(n=n, MTOM=MTOM_best, l_f=l_f, b=b_wing_best)
    mass_wing_best = M_wing(b=b_wing_best, c=c_best, sweep=sweep, taper_ratio=taper_ratio,
                            t_over_c=t_over_c, n=n, MTOM=MTOM_best, V_cruise=V_cruise_best)
    mass_payload_best = M_payload(M_pax=m_pax, M_lug=m_lugg, n_seats=n_seats)
    mass_engine_best = M_engine(T_to_W=T_to_W, T_cruise=T_SL_best)
    mass_fuselage_best = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM_best, V_cruise=V_cruise_best)
    mass_furnish_best = M_furnish(MTOM=MTOM_best, n_crew=n_crew, V_cruise=V_cruise_best, b=b_wing_best, c=c_best)
    mass_fuel_sys_best = mass_fuel_best / rho_fuel_sys
    mass_battery_best = M_battery(E_trip=energy_battery_best, E_reserve=0, rho_bat=rho_batt, e_usable=0.80)

    mass_total_best = (mass_motor_best + mass_rotor_best + mass_systems_best + mass_wing_best + mass_payload_best +
                       mass_engine_best + mass_fuselage_best + mass_furnish_best + mass_fuel_sys_best + mass_battery_best)

    # Economic analysis at best point
    mass_empty_best = max(mass_total_best - (mass_payload_best + mass_fuel_sys_best + mass_battery_best), 0.0)
    TOC_best, profit_trip_best, profit_annual_best = economic_analysis(
        n_wd=n_wd,
        T_D=T_D,
        C_rate_charge=C_rate_charge,
        E_trip=energy_trip_best / 1000,           # kWh
        E_battery=energy_battery_best / 1000,     # kWh
        t_trip=time_trip_best,                    # hours
        P_energy=P_energy,
        j=j,
        P_empty=P_empty,
        M_empty=mass_empty_best,
        x_ins=x_ins,
        x_irs=x_irs,
        x_iap=x_iap,
        x_iga=x_iga,
        MF=MF,
        MWR=MWR,
        P_batt=P_batt,
        rho_batt=rho_batt,
        M_batt=mass_battery_best,
        N_cycles=N_cycles,
        D_trip=range_cruise / 1000,               # km
        n_ac=n_ac,
        P_pilot=P_pilot,
        fare=fare
    )

    print("\n### --- Display Results --- ###")
    print("--- Optimization Variables ---")
    print(f"Optimal V_cruise: {V_cruise_best * 3.6:.2f} kph")
    print(f"Optimal V_climb: {V_climb_best * 3.6:.2f} kph")
    print(f"Optimal MTOM: {MTOM_best:.2f} kg")
    print(f"Optimal Disk Area: {A_disk_best:.3f} m^2")
    print(f"Optimal Wing Area: {S_best:.3f} m^2")
    print(f"Optimal Wing Aspect Ratio: {AR_best:.3f}")

    print("\n--- Wing Planform Properties ---")
    print(f"Wingspan: {b_wing_best:.3f} m")
    print(f"Wing chord length: {c_best:.3f} m")
    print(f"Wing loading: {wing_loading_best:.1f} N/m^2")

    print("\n--- Aerodynamic Properties ---")
    print(f"CL at cruise: {CL_cruise_best:.5f}")
    print(f"CL/CD at cruise: {CL_over_CD_cruise_best:.3f}")
    print(f"CL at climb: {CL_climb_best:.5f}")
    print(f"CL/CD at climb: {CL_over_CD_climb_best:.3f}")

    print("\n--- Thrust and Power Properties ---")
    print(f"Thrust available at sea level for cruise: {T_SL_cruise_best:.1f} N")
    print(f"Thrust available at sea level for climb: {T_SL_climb_best:.1f} N")
    print(f"Thrust available at sea level: {T_SL_best:.1f} N")
    print(f"Thrust-to-weight ratio: {T_SL_best / (MTOM_best * g):.3f}")
    print(f"Total hover power required: {P_hover_best / 1000:.2f} kW")
    print(f"Motor power required: {(P_hover_best / 1000) / num_rotors} kW")
    print(f"Power loading: {(P_hover_best / 1000) / (MTOM_best * g)} kW/N")
    print(f"Disk loading: {(MTOM_best * g) / (A_disk_best)} N/m^2")

    print("\n--- Energy Properties ---")
    print(f"Battery energy required: {energy_battery_best / 1000:.3f} kWh")
    print(f"Fuel energy required: {energy_fuel_best / 1000:.3f} kWh")
    print(f"Total trip energy required: {energy_trip_best / 1000:.3f} kWh")

    print("\n--- Time Properties ---")
    print(f"Total cruise time: {time_cruise_best:.4f} hours")
    print(f"Total climb time: {time_climb_best:.4f} hours")
    print(f"Total trip time: {time_trip_best:.4f} hours")

    print("\n--- Mass Properties ---")
    print(f"Total motor mass: {mass_motor_best:.3f} kg")
    print(f"Motor mass: {mass_motor_best/num_rotors:.3f} kg")
    print(f"Total rotor mass: {mass_rotor_best:.3f} kg")
    print(f"Total systems mass: {mass_systems_best:.3f} kg")
    print(f"Total payload mass: {mass_payload_best:.3f} kg")
    print(f"Total wing mass: {mass_wing_best:.3f} kg")
    print(f"Total fuselage mass: {mass_fuselage_best:.3f} kg")
    print(f"Total furnishing mass: {mass_furnish_best:.3f} kg")
    print(f"Total battery mass: {mass_battery_best:.3f} kg")
    print(f"Total engine mass: {mass_engine_best:.3f} kg")
    print(f"Total fuel mass: {mass_fuel_best:.3f} kg")
    print(f"Total fuel system mass: {mass_fuel_sys_best:.3f} kg")
    print(f"Total fuel system volume: {mass_fuel_sys_best / rho_fuel_vol:.3f} L")
    print(f"Maximum takeoff mass: {mass_total_best:.3f} kg")

    print("\n--- Economic Properties ---")
    print(f"Total operating cost: {TOC_best:.3f} $")
    print(f"Total profit per trip: {profit_trip_best:.3f} $")
    print(f"Total profit per annum: {profit_annual_best:.3f} $")



if __name__ == "__main__":
    main()
