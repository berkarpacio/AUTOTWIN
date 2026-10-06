def small_UAV_eVTOL_MOO(input_dict):
    """
    Single-objective NSGA-II optimization for small LH2-powered eVTOL UAV trip sizing.

    Decision vector:
        x = [MTOM, S, V_cruise, AR, R_cruise, R_hover]

    Objectives (minimize):
        F = [energy_trip_Wh]

    Constraints (feasible if G <= 0):
        1) MTOM <= MTOM_max
        2) V_cruise >= V_min
        3) (V_cruise - V_stall) >= stall_margin
        4) b_wing <= b_wing_max
        5) |MTOM - mass_total| <= tol_mass_kg

    Inputs:
        input_dict: dict of parameters

    Returns:
        output_dict: dict
    """


    # --- 0) Imports ---
    import math
    import numpy as np
    from ambiance import Atmosphere

    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    # Import Physics Models
    from libraries.aerospace.optimization.models.Power_models import power_hover, power_cruise
    from libraries.aerospace.optimization.models.Mass_models import M_wing, M_fuselage, M_battery, M_motor


    # --- 1) Read inputs ---
    # Aircraft
    t_hover = input_dict["t_hover"] / 60.0  # minutes -> hours
    t_loiter = input_dict["t_loiter"] / 60.0 # minutes -> hours
    CD0 = float(input_dict["CD0"])
    CL_max = float(input_dict["CL_max"])
    stall_margin = float(input_dict["stall_margin"])  # [m/s]

    # Wing
    n = float(input_dict["n_wing"])
    sweep = math.radians(float(input_dict["theta_wing"]))
    taper_ratio = float(input_dict["taper_ratio_wing"])
    t_over_c = float(input_dict["t_over_c_wing"])
    e = float(input_dict["e_wing"])

    # Fuselage
    l_f = float(input_dict["length_fuselage"])
    w_f = float(input_dict["width_fuselage"])
    d_f = float(input_dict["depth_fuselage"])

    # Battery
    rho_batt = float(input_dict["rho_battery"]) # [Wh/kg]
    e_usable = float(input_dict["e_usable_battery"])
    hybrid_batt_factor = float(input_dict["hybrid_batt_factor"])

    # Fuel cell 
    eta_fc = float(input_dict["eta_fc"])
    rho_fc = float(input_dict["rho_fc"]) # [W/kg]
    hybrid_fc_factor = float(input_dict["hybrid_fc_factor"])

    # LH2 storage
    gravimetric_index = float(input_dict["gravimetric_index"])
    LHV_H2 = float(input_dict["LHV_H2"]) # [Wh/kg]

    # Hover BLDC
    eta_elec = float(input_dict["eta_elec_hover_bldc"])
    rho_motor = float(input_dict["rho_motor_hover_bldc"])  # [W/kg]
    num_rotors_hover = int(input_dict["num_rotors_hover_propeller"])
    eta_prop_hover = float(input_dict["eta_prop_hover_propeller"])
    eta_hover = eta_elec * eta_prop_hover

    # Cruise BLDC
    num_rotors_cruise = int(input_dict["num_rotors_cruise_propeller"])
    eta_prop_cruise = float(input_dict["eta_prop_cruise_propeller"])
    eta_cruise = eta_elec * eta_prop_cruise

    # Sizing mission
    b_cruise = float(input_dict["b_cruise"])
    h_cruise = float(input_dict["h_cruise"])
    h_hover = float(input_dict["h_hover"])
    rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
    rho_hover = float(np.squeeze(Atmosphere(h_hover).density))
    g = float(input_dict["g"])

    # Mass constants
    m_payload = float(input_dict["mass_payload"])
    m_flight_controller = float(input_dict["mass_flight_controller"])
    m_flight_computer = float(input_dict["mass_flight_computer"])
    m_avionics = float(input_dict["mass_avionics"])
    m_electronics = float(input_dict["mass_electronics"])
    num_servo = int(input_dict["num_servo"])
    m_servo = float(input_dict["mass_servo"])
    m_servos = m_servo * num_servo

    # Constraints
    MTOM_lb = float(input_dict["MTOM_bounds"][0])
    MTOM_ub = float(input_dict["MTOM_bounds"][1])

    V_cruise_lb = float(input_dict["V_cruise_bounds"][0])
    V_cruise_ub = float(input_dict["V_cruise_bounds"][1])

    S_lb = float(input_dict["S_wing_bounds"][0])
    S_ub = float(input_dict["S_wing_bounds"][1])

    AR_lb = float(input_dict["AR_wing_bounds"][0])
    AR_ub = float(input_dict["AR_wing_bounds"][1])

    R_cruise_lb = float(input_dict["radius_cruise_propeller_bounds"][0])
    R_cruise_ub = float(input_dict["radius_cruise_propeller_bounds"][1])

    R_hover_lb = float(input_dict["radius_hover_propeller_bounds"][0])
    R_hover_ub = float(input_dict["radius_hover_propeller_bounds"][1])

    MTOM_max = MTOM_ub
    b_wing_max = float(input_dict["b_wing_bounds"][1])
    V_min_cruise = V_cruise_lb
    tol_mass_kg = float(input_dict["tol_mass_kg"])

    # NSGA-II settings
    pop_size = 200
    n_gen = 200
    seed = 1
    verbose = True


    # --- 2) Numeric evaluator ---
    def evaluate_trip(x: np.ndarray):
        """
        x = [MTOM, S, V_cruise, AR, R_cruise, R_hover]
        Returns: (F, G, debug)
        """
        MTOM, S, V_cruise, AR, R_cruise, R_hover = [float(v) for v in x]

        # Geometry
        b_wing = math.sqrt(max(S * AR, 1e-12))
        c = S / b_wing if b_wing > 0 else 1e30
        wing_loading = (MTOM * g) / max(S, 1e-12)

        # Aero
        K = 1.0 / (math.pi * e * max(AR, 1e-12))
        CL_cruise = (2.0 * b_cruise * MTOM * g) / (rho_cruise * max(S, 1e-12) * max(V_cruise, 1e-12) ** 2)
        CD_cruise = CD0 + K * CL_cruise**2
        LD_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else float("inf")
        V_stall = math.sqrt((2.0 * MTOM * g) / (CL_max * rho_cruise * max(S, 1e-12)))

        # Power (W)
        P_hover = power_hover(
            MTOM=1.15 * MTOM,
            rho=rho_hover,
            R_prop=R_hover,
            num_rotors=num_rotors_hover,
            eta_h=eta_hover,
        )
        P_cruise = power_cruise(
            MTOM=MTOM,
            b_cruise=b_cruise,
            AR=AR,
            e=e,
            CD0=CD0,
            rho_cruise=rho_cruise,
            S=S,
            V_cruise=V_cruise,
            R_cruise=R_cruise,
            eta_cruise=eta_cruise,
            num_rotors_cruise=num_rotors_cruise,
        )

        # Fuel cell
        P_fc = max(P_cruise, P_hover)

        # LH2
        rH2_hover = (P_hover * hybrid_fc_factor) / (eta_fc * LHV_H2) # [kg/h]
        rH2_cruise = P_cruise / (eta_fc * LHV_H2) # [kg/h]
        rH2_max = max(rH2_cruise, rH2_hover) # [kg/h]

        # Battery
        energy_battery = P_hover * hybrid_batt_factor * t_hover

        # Mass models (kg)
        mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
        mass_motor_cruise = M_motor(P_out=P_cruise, rho_motor=rho_motor)
        mass_motor_hover = M_motor(P_out=P_hover, rho_motor=rho_motor)
        mass_motor = mass_motor_cruise + mass_motor_hover
        mass_fuel_system = (rH2_hover * t_hover + rH2_cruise * t_loiter) * (1/gravimetric_index)
        mass_fuel_cell = P_fc / rho_fc

        mass_wing = M_wing(
            b=b_wing,
            c=c,
            sweep=sweep,
            taper_ratio=taper_ratio,
            t_over_c=t_over_c,
            n=n,
            MTOM=MTOM,
            V_cruise=V_cruise,
        )
        mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
        mass_components = m_avionics + m_electronics + m_flight_controller + m_flight_computer + m_servos

        mass_total = (
            mass_motor
            + mass_wing
            + m_payload
            + mass_fuselage
            + mass_battery
            + mass_fuel_cell
            + mass_fuel_system
            + mass_components
        )

        # Total trip energy
        energy_trip = (energy_battery / e_usable) + (mass_fuel_system * gravimetric_index) * LHV_H2

        # Constraints (<= 0 feasible)
        G = np.array(
            [
                MTOM - MTOM_max,
                V_min_cruise - V_cruise,
                stall_margin - (V_cruise - V_stall),
                b_wing - b_wing_max,
                abs(MTOM - mass_total) - tol_mass_kg,
            ],
            dtype=float,
        )

        # Objectives
        F = np.array([energy_trip], dtype=float)

        # Numerical guards
        if (not np.all(np.isfinite(F))) or (not np.all(np.isfinite(G))):
            F = np.array([1e30], dtype=float)
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
            "V_stall": V_stall,
            "R_cruise": R_cruise,
            "R_hover": R_hover,
            "CL_cruise": CL_cruise,
            "CD_cruise": CD_cruise,
            "LD_cruise": LD_cruise,
            "P_hover": P_hover,
            "P_cruise": P_cruise,
            "energy_trip": energy_trip,
            "mass_components": mass_components,
            "mass_battery": mass_battery,
            "mass_fuselage": mass_fuselage,
            "mass_wing": mass_wing,
            "mass_motor_hover": mass_motor_hover,
            "mass_motor_cruise": mass_motor_cruise,
            "mass_fuel_system": mass_fuel_system,
            "mass_fuel_cell": mass_fuel_cell,
            "hover_motor_mass_per_rotor": (mass_motor_hover / num_rotors_hover) if num_rotors_hover > 0 else float("inf"),
            "cruise_motor_mass_per_rotor": (mass_motor_cruise / num_rotors_cruise) if num_rotors_cruise > 0 else float("inf"),
        }

        return F, G, debug

    
    # --- 3) pymoo Problem wrapper ---
    class EVTOLUAVTripProblem(Problem):
        def __init__(self):
            xl = np.array([
                    MTOM_lb,     # MTOM [kg]
                    S_lb,        # S [m^2]
                    V_cruise_lb, # V_cruise [m/s]
                    AR_lb,       # AR [-]
                    R_cruise_lb, # R_cruise [m]
                    R_hover_lb,  # R_hover  [m]
                ],
                dtype=float,
            )
            xu = np.array(
                [
                    MTOM_ub,     # MTOM [kg]
                    S_ub,        # S [m^2]
                    V_cruise_ub, # V_cruise [m/s]
                    AR_ub,       # AR [-]
                    R_cruise_ub, # R_cruise [m]
                    R_hover_ub,  # R_hover  [m]
                ],
                dtype=float,
            )
            super().__init__(n_var=6, n_obj=1, n_ieq_constr=5, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 1), dtype=float)
            G = np.zeros((X.shape[0], 5), dtype=float)
            for i in range(X.shape[0]):
                f, g, _ = evaluate_trip(X[i, :])
                F[i, :] = f
                G[i, :] = g
            out["F"] = F
            out["G"] = G


    # --- 4) Run NSGA-II ---
    problem = EVTOLUAVTripProblem()
    algorithm = NSGA2(pop_size=pop_size, eliminate_duplicates=True)

    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
    )

    # If no solutions returned
    if res.X is None or res.F is None:
        return {
            "status": "no_solution_found",
            "message": "Optimizer returned no solution. Relax constraints or increase generations.",
        }

    X_sol = np.atleast_2d(res.X)
    F_sol = np.atleast_2d(res.F)

    best_idx = int(np.argmin(F_sol[:, 0]))
    x_best = X_sol[best_idx]
    f_best = F_sol[best_idx]
    _, g_best, dbg_best = evaluate_trip(x_best)

    cv = float(np.max(np.maximum(0.0, g_best)))

    output_dict = {
        "MTOM": dbg_best["MTOM"],
        "mass_total": dbg_best["mass_total"],
        "S_wing": dbg_best["S"],
        "AR_wing": dbg_best["AR"],
        "b_wing": dbg_best["b_wing"],
        "V_cruise": dbg_best["V_cruise"],
        "V_stall": dbg_best["V_stall"],
        "radius_cruise_propeller": dbg_best["R_cruise"],
        "radius_hover_propeller": dbg_best["R_hover"],
        "e_trip": dbg_best["energy_trip"],
        "CL_cruise": dbg_best["CL_cruise"],
        "CD_cruise": dbg_best["CD_cruise"],
        "CL_over_CD_cruise": dbg_best["LD_cruise"],
        "P_hover": dbg_best["P_hover"],
        "P_cruise": dbg_best["P_cruise"],
        "P_motor_hover_bldc": dbg_best["P_hover"] / max(num_rotors_hover, 1),
        "P_motor_cruise_bldc": dbg_best["P_cruise"] / max(num_rotors_cruise, 1),
        "mass_components": dbg_best["mass_components"],
        "mass_battery": dbg_best["mass_battery"],
        "mass_fuselage": dbg_best["mass_fuselage"],
        "mass_wing": dbg_best["mass_wing"],
        "mass_motor_hover": dbg_best["mass_motor_hover"],
        "mass_motor_cruise": dbg_best["mass_motor_cruise"],
        "mass_hover_bldc_per_rotor": dbg_best["hover_motor_mass_per_rotor"],
        "mass_cruise_bldc_per_rotor": dbg_best["cruise_motor_mass_per_rotor"],
        "mass_fuel_system": dbg_best["mass_fuel_system"],
        "mass_fuel_cell": dbg_best["mass_fuel_cell"],
        "wing_loading": (dbg_best["MTOM"] * g) / max(dbg_best["S"], 1e-12),
        "disk_loading": (dbg_best["MTOM"] * g) / (max(num_rotors_hover, 1) * math.pi * max(dbg_best["R_hover"], 1e-12) ** 2),
        "status": "ok" if cv <= 0 else "infeasible_best_found",
        "best_idx": best_idx,
        "best_objectives": {"energy_trip_Wh": float(f_best[0])},
        "max_constraint_violation": cv,
        "solution_X": X_sol,
        "solution_F": F_sol,
    }

    return output_dict

input_dict = {
    # Mission
    "t_hover": 10.0,                   # [min] total hover time
    "t_loiter": 1900,                  # [min] cruise/loiter segment used by current model

    # Aerodynamics
    "CD0": 0.035,
    "CL_max": 1.6,
    "stall_margin": 5.0,               # [m/s]

    # Wing
    "n_wing": 3.5,                     # ultimate load factor / structural factor used by your mass model
    "theta_wing": 0.0,                 # [deg]
    "taper_ratio_wing": 0.6,
    "t_over_c_wing": 0.12,
    "e_wing": 0.80,

    # Fuselage
    "length_fuselage": 1.40,           # [m]
    "width_fuselage": 0.22,            # [m]
    "depth_fuselage": 0.18,            # [m]

    # Battery
    "rho_battery": 180.0,              # [Wh/kg]
    "e_usable_battery": 0.85,          # [-]
    "hybrid_batt_factor": 0.10,        # [-] fraction of hover power supplied by battery

    # Fuel cell
    "eta_fc": 0.50,                    # [-]
    "rho_fc": 840.0,                   # [W/kg] specific power of FC system
    "hybrid_fc_factor": 0.90,          # [-] fraction of hover power supplied by FC

    # LH2 storage
    "gravimetric_index": 0.18,         # [-] hydrogen mass / total tank-system mass
    "LHV_H2": 33330.0,                 # [Wh/kg]

    # Hover propulsion
    "eta_elec_hover_bldc": 0.95,       # [-]
    "rho_motor_hover_bldc": 3000.0,    # [W/kg]
    "num_rotors_hover_propeller": 8,
    "eta_prop_hover_propeller": 0.75,  # [-]

    # Cruise propulsion
    "num_rotors_cruise_propeller": 1,
    "eta_prop_cruise_propeller": 0.82, # [-]

    # Flight condition
    "b_cruise": 1.0,                   # load factor / correction factor used in your CL expression
    "h_cruise": 500.0,                 # [m]
    "h_hover": 0.0,                    # [m]
    "g": 9.81,                         # [m/s^2]

    # Fixed masses
    "mass_payload": 3.0,               # [kg]
    "mass_flight_controller": 0.12,    # [kg]
    "mass_flight_computer": 0.18,      # [kg]
    "mass_avionics": 0.5,             # [kg]
    "mass_electronics": 0.5,          # [kg]
    "num_servo": 5,
    "mass_servo": 0.06,                # [kg]

    # Design variable bounds
    "MTOM_bounds": [8.0, 25.0],                        # [kg]
    "V_cruise_bounds": [20.0, 40.0],                  # [m/s]
    "S_wing_bounds": [0.35, 20],                    # [m^2]
    "AR_wing_bounds": [6.0, 12.0],                    # [-]
    "radius_cruise_propeller_bounds": [2*0.0254, 5*0.0254],  # [m]
    "radius_hover_propeller_bounds": [2*0.0254, 5*0.0254],   # [m]

    # Constraints
    "b_wing_bounds": [0.0, 5.0],       # [m]
    "tol_mass_kg": 0.05                # [kg]
}

# Example run
result = small_UAV_eVTOL_MOO(input_dict)

print("Status:", result["status"])
print("Best objective:", result["best_objectives"])
print("MTOM [kg]:", result["MTOM"])
print("Total mass [kg]:", result["mass_total"])
print("Fuel system mass:", result["mass_fuel_system"])
print("Fuel cell mass:", result["mass_fuel_cell"])
print("Battery mass:", result["mass_battery"])
print("Wing mass:", result["mass_wing"])
print("Total hover motor mass:", result["mass_motor_hover"])
print("Total cruise motor mass:", result["mass_motor_cruise"])
print("Fuselage mass:", result["mass_fuselage"])
print("Total component mass:", result["mass_components"])
print("Wing area [m^2]:", result["S_wing"])
print("Cruise speed [m/s]:", result["V_cruise"])
print("Aspect ratio:", result["AR_wing"])
print("R_cruise:", result["radius_cruise_propeller"])
print("R_hover:", result["radius_hover_propeller"])
print("Max constraint violation:", result["max_constraint_violation"])
