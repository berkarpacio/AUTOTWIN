def small_UAV_eVTOL_MOO(input_dict):
    """
    Single-objective NSGA-II optimization for small GH2-powered MULTIROTOR UAV trip sizing.

    Decision vector:
        x = [MTOM, R_prop]

    Objectives (minimize):
        F = [energy_trip_Wh]

    Constraints (feasible if G <= 0):
        1) MTOM <= MTOM_max
        2) |MTOM - mass_total| <= tol_mass_kg

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
    from libraries.aerospace.models.optimization.models.Power_models import power_hover
    from libraries.aerospace.models.optimization.models.Mass_models import M_battery, M_motor, V_hydrogen


    # --- 1) Read inputs ---
    # Aircraft
    t_hover = input_dict["t_hover"] / 60.0  # minutes -> hours

    # Battery
    rho_batt = float(input_dict["rho_battery"]) # [Wh/kg]
    e_usable = float(input_dict["e_usable_battery"])
    hybrid_batt_factor = float(input_dict["hybrid_batt_factor"])

    # Fuel cell 
    eta_fc = float(input_dict["eta_fc"])
    rho_fc = float(input_dict["rho_fc"]) # [W/kg]
    hybrid_fc_factor = float(input_dict["hybrid_fc_factor"])

    # GH2 storage
    gravimetric_index = float(input_dict["gravimetric_index"])
    LHV_H2 = float(input_dict["LHV_H2"]) # [Wh/kg]

    # Hover BLDC
    eta_elec = float(input_dict["eta_elec_hover_bldc"])
    rho_motor = float(input_dict["rho_motor_hover_bldc"])  # [W/kg]
    num_rotors_hover = int(input_dict["num_rotors_hover_propeller"])
    eta_prop_hover = float(input_dict["eta_prop_hover_propeller"])
    eta_hover = eta_elec * eta_prop_hover

    # Sizing mission
    h_hover = float(input_dict["h_hover"])
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

    R_hover_lb = float(input_dict["radius_hover_propeller_bounds"][0])
    R_hover_ub = float(input_dict["radius_hover_propeller_bounds"][1])

    MTOM_max = MTOM_ub
    tol_mass_kg = float(input_dict["tol_mass_kg"])

    # NSGA-II settings
    pop_size = 200
    n_gen = 200
    seed = 1
    verbose = True


    # --- 2) Numeric evaluator ---
    def evaluate_trip(x: np.ndarray):
        """
        x = [MTOM, R_hover]
        Returns: (F, G, debug)
        """
        MTOM, R_hover = [float(v) for v in x]

        # Power (W)
        P_hover = power_hover(
            MTOM=MTOM,
            rho=rho_hover,
            R_prop=R_hover,
            num_rotors=num_rotors_hover,
            eta_h=eta_hover,
        )

        # Fuel cell
        P_fc = 1.15*P_hover*hybrid_fc_factor

        # GH2
        rH2_hover = (P_hover * hybrid_fc_factor) / (eta_fc * LHV_H2) # [kg/h]
        m_fuel = rH2_hover * t_hover

        # Battery
        energy_battery = P_hover * hybrid_batt_factor * t_hover

        # Mass models (kg)
        mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
        mass_motor_hover = M_motor(P_out=P_hover, rho_motor=rho_motor)
        mass_motor = mass_motor_hover
        mass_fuel_system = (rH2_hover * t_hover) * (1/gravimetric_index)
        mass_fuel_cell = P_fc / rho_fc
        mass_airframe = MTOM * 0.25

        mass_components = m_avionics + m_electronics + m_flight_controller + m_flight_computer + m_servos

        mass_total = (
            mass_motor
            + m_payload
            + mass_battery
            + mass_fuel_cell
            + mass_fuel_system
            + mass_components
            + mass_airframe
        )

        # GH2 Tank Volume Model
        v_tank_m3 = V_hydrogen(Z=1.236, R=8.314, T=293, m_H2_stored=m_fuel, P_tank=35e6)
        v_tank_L = v_tank_m3 * 1000


        # Total trip energy
        energy_trip = (energy_battery / e_usable) + (mass_fuel_system * gravimetric_index) * LHV_H2

        # Constraints (<= 0 feasible)
        G = np.array(
            [
                MTOM - MTOM_max,
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
            "R_hover": R_hover,
            "P_hover": P_hover,
            "energy_trip": energy_trip,
            "mass_components": mass_components,
            "mass_battery": mass_battery,
            "mass_motor_hover": mass_motor_hover,
            "mass_fuel": m_fuel,
            "v_tank": v_tank_L,
            "mass_fuel_system": mass_fuel_system,
            "mass_fuel_cell": mass_fuel_cell,
            "hover_motor_mass_per_rotor": (mass_motor_hover / num_rotors_hover) if num_rotors_hover > 0 else float("inf")
        }

        return F, G, debug

    
    # --- 3) pymoo Problem wrapper ---
    class EVTOLUAVTripProblem(Problem):
        def __init__(self):
            xl = np.array([
                    MTOM_lb,     # MTOM [kg]
                    R_hover_lb,  # R_hover  [m]
                ],
                dtype=float,
            )
            xu = np.array(
                [
                    MTOM_ub,     # MTOM [kg]
                    R_hover_ub,  # R_hover  [m]
                ],
                dtype=float,
            )
            super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 1), dtype=float)
            G = np.zeros((X.shape[0], 2), dtype=float)
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
        "radius_hover_propeller": dbg_best["R_hover"],
        "e_trip": dbg_best["energy_trip"],
        "P_hover": dbg_best["P_hover"],
        "P_motor_hover_bldc": dbg_best["P_hover"] / max(num_rotors_hover, 1),
        "mass_components": dbg_best["mass_components"],
        "mass_battery": dbg_best["mass_battery"],
        "mass_motor_hover": dbg_best["mass_motor_hover"],
        "mass_hover_bldc_per_rotor": dbg_best["hover_motor_mass_per_rotor"],
        "mass_fuel_system": dbg_best["mass_fuel_system"],
        "mass_fuel": dbg_best["mass_fuel"],
        "v_tank": dbg_best["v_tank"],
        "mass_fuel_cell": dbg_best["mass_fuel_cell"],
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
    "t_hover": 90.0,                   # [min] total hover time

    # Battery
    "rho_battery": 180.0,              # [Wh/kg]
    "e_usable_battery": 0.85,          # [-]
    "hybrid_batt_factor": 0.05,        # [-] fraction of hover power supplied by battery

    # Fuel cell
    "eta_fc": 0.50,                    # [-]
    "rho_fc": 840.0,                   # [W/kg] specific power of FC system
    "hybrid_fc_factor": 0.95,          # [-] fraction of hover power supplied by FC

    # GH2 storage
    "gravimetric_index": 0.06,         # [-] hydrogen mass / total tank-system mass
    "LHV_H2": 33330.0,                 # [Wh/kg]

    # Hover propulsion
    "eta_elec_hover_bldc": 0.95,       # [-]
    "rho_motor_hover_bldc": 3000.0,    # [W/kg]
    "num_rotors_hover_propeller": 4,
    "eta_prop_hover_propeller": 0.75,  # [-]

    # Flight condition
    "h_hover": 1000.0,                    # [m]
    "g": 9.81,                         # [m/s^2]

    # Fixed masses
    "mass_payload": 3.0,               # [kg]
    "mass_flight_controller": 0.12,    # [kg]
    "mass_flight_computer": 0.18,      # [kg]
    "mass_avionics": 0.5,             # [kg]
    "mass_electronics": 0.5,          # [kg]
    "num_servo": 0,
    "mass_servo": 0.06,                # [kg]

    # Design variable bounds
    "MTOM_bounds": [1.0, 25.0],                               # [kg]
    "radius_hover_propeller_bounds": [2*0.0254, 10*0.0254],   # [m]

    # Constraints
    "tol_mass_kg": 0.05                # [kg]
}

# Example run
result = small_UAV_eVTOL_MOO(input_dict)

print("Status:", result["status"])
print("Best objective:", result["best_objectives"])
print("MTOM [kg]:", result["MTOM"])
print("Total mass [kg]:", result["mass_total"])
print("Fuel system mass:", result["mass_fuel_system"])
print("Fuel mass:", result["mass_fuel"])
print("Tank volume:", result["v_tank"])
print("Fuel cell mass:", result["mass_fuel_cell"])
print("Battery mass:", result["mass_battery"])
print("Total hover motor mass:", result["mass_motor_hover"])
print("Total component mass:", result["mass_components"])
print("R_hover:", result["radius_hover_propeller"])
print("Max constraint violation:", result["max_constraint_violation"])
