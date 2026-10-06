def small_UAV_eVTOL_MOO(input_dict):
    """
    Multi-objective NSGA-II optimization for small eVTOL UAV trip sizing.

    Decision vector:
        x = [MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover]

    Objectives (minimize):
        F = [energy_stored_onboard_Wh, t_trip_h]

    Constraints (feasible if G <= 0):
        1) MTOM <= MTOM_max
        2) V_cruise >= V_min
        3) V_climb >= V_min
        4) (V_cruise - V_stall) >= stall_margin
        5) b_wing <= b_wing_max
        6) |MTOM - mass_total| <= tol_mass_kg

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
    from libraries.aerospace.optimization.models.Power_models import power_hover, power_cruise, power_climb
    from libraries.aerospace.optimization.models.Mass_models import M_battery, M_motor, V_hydrogen


    # --- 1) Read inputs ---
    # Aircraft
    t_hover = input_dict["t_hover"] / 60.0  # minutes -> hours
    range_cruise = input_dict["range_cruise"] * 1e3  # km -> m
    CD0 = float(input_dict["CD0"])
    CL_max = float(input_dict["CL_max"])
    stall_margin = float(input_dict["stall_margin"])  # [m/s]
    e = float(input_dict["e_wing"])
    l_max = float(input_dict["l_max"])

    # Battery
    rho_batt = float(input_dict["rho_battery"])  # [Wh/kg]
    e_usable = float(input_dict["e_usable_battery"])

    # Fuel cell 
    eta_fc = float(input_dict["eta_fc"])  # [-]
    rho_fc = float(input_dict["rho_fc"])  # [-]

    # GH2
    GI = float(input_dict["gravimetric_index"])     
    LHV_H2 = float(input_dict["LHV_H2"])   
    tank_AR = float(input_dict["tank_AR"])  
    Z_tank = float(input_dict["Z_tank"]) 
    P_tank = float(input_dict["P_tank"])   
    T_tank = float(input_dict["T_tank"])  

    # Hover BLDC
    eta_elec_hover = float(input_dict["eta_elec_hover_bldc"])
    rho_motor_hover = float(input_dict["rho_motor_hover_bldc"])  # [W/kg]
    num_rotors_hover = int(input_dict["num_rotors_hover_propeller"])
    eta_prop_hover = float(input_dict["eta_prop_hover_propeller"])
    eta_hover = eta_elec_hover * eta_prop_hover

    # Cruise BLDC
    eta_elec_cruise = float(input_dict["eta_elec_cruise_bldc"])
    rho_motor_cruise = float(input_dict["rho_motor_cruise_bldc"])  # [W/kg]
    num_rotors_cruise = int(input_dict["num_rotors_cruise_propeller"])
    eta_prop_cruise = float(input_dict["eta_prop_cruise_propeller"])
    eta_cruise = eta_elec_cruise * eta_prop_cruise

    # Sizing mission
    b_cruise = float(input_dict["b_cruise"])
    b_climb = float(input_dict["b_climb"])
    h_cruise = float(input_dict["h_cruise"])
    h_climb = float(input_dict["h_climb_start"])
    theta_climb = math.radians(float(input_dict["theta_climb"]))  # deg -> rad
    h_hover = float(input_dict["h_hover"])
    rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
    rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
    rho_hover = float(np.squeeze(Atmosphere(h_hover).density))
    n_batt_hover = float(input_dict["hybrid_batt_factor_hover"])
    n_batt_cruise = float(input_dict["hybrid_batt_factor_cruise"])
    n_batt_climb = float(input_dict["hybrid_batt_factor_climb"])
    g = float(input_dict["g"])
    vertical_climb_factor = float(input_dict["vertical_climb_factor"])

    # Mass constants
    m_payload = float(input_dict["mass_payload"])
    m_flight_controller = float(input_dict["mass_flight_controller"])
    m_flight_computer = float(input_dict["mass_flight_computer"])
    num_servo = int(input_dict["num_servo"])
    m_servo = float(input_dict["mass_servo"])
    m_servos = m_servo * num_servo
    mass_auxilliary = float(input_dict["mass_auxilliary"])
    m_structures_factor = float(input_dict["mass_structures_factor"])

    # Constraints
    MTOM_lb = float(input_dict["MTOM_bounds"][0])
    MTOM_ub = float(input_dict["MTOM_bounds"][1])

    V_cruise_lb = float(input_dict["V_cruise_bounds"][0])
    V_cruise_ub = float(input_dict["V_cruise_bounds"][1])

    V_climb_lb = float(input_dict["V_climb_bounds"][0])
    V_climb_ub = float(input_dict["V_climb_bounds"][1])

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
    V_min_climb = V_climb_lb
    tol_mass_kg = float(input_dict["tol_mass_kg"])

    # NSGA-II settings
    pop_size = int(input_dict.get("pop_size", 200))
    n_gen = int(input_dict.get("n_gen", 200))
    seed = int(input_dict.get("seed", 1))
    verbose = bool(input_dict.get("verbose", True))
    plot_pareto = bool(input_dict.get("plot_pareto", True))
    pareto_plot_path = input_dict.get("pareto_plot_path")
    pareto_plot_block = bool(input_dict.get("pareto_plot_block", pareto_plot_path is None))

    # Compromise selection weights
    wE = 0.5
    wT = 0.5


    # --- 2) Numeric evaluator ---
    def evaluate_trip(x: np.ndarray):
        """
        x = [MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover]
        Returns: (F, G, debug)
        """
        MTOM, S, V_cruise, V_climb, AR, R_cruise, R_hover = [float(v) for v in x]

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

        # Power in required at ESC in terminals (W)
        P_hover = power_hover(
            MTOM=vertical_climb_factor*MTOM,
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
        P_climb = power_climb(
            MTOM=MTOM,
            b_climb=b_climb,
            AR=AR,
            e=e,
            CD0=CD0,
            rho_climb=rho_climb,
            S=S,
            V_climb=V_climb,
            theta_climb=theta_climb,
            R_cruise=R_cruise,
            eta_cruise=eta_cruise,
            num_rotors_cruise=num_rotors_cruise,
        )

        # Hover motor
        P_hover_motor_max = P_hover * eta_elec_hover # shaft power for hover motor (W)
        P_cruise_motor_max = max(P_cruise, P_climb) * eta_elec_cruise # shaft power for cruise motor (W)

        # Fuel cell
        P_fc_hover = (1 - n_batt_hover) * P_hover
        P_fc_climb = (1 - n_batt_climb) * P_climb
        P_fc_cruise = (1 - n_batt_cruise) * P_cruise
        P_fc = max(P_fc_climb, P_fc_cruise, P_fc_hover)

        # Battery 
        P_batt_hover = n_batt_hover * P_hover
        P_batt_climb = n_batt_climb * P_climb
        P_batt_cruise = n_batt_cruise * P_cruise

        # Time (hours)
        t_cruise = (range_cruise / max(V_cruise, 1e-12)) / 3600.0
        t_climb = ((h_cruise - h_climb) / max(V_climb * math.sin(theta_climb), 1e-12)) / 3600.0
        t_trip = t_cruise + t_climb + t_hover

        # Battery energy (Wh) since P[W]*t[h] = Wh
        energy_battery_hover = P_batt_hover * t_hover
        energy_battery_cruise = P_batt_cruise * t_cruise
        energy_battery_climb = P_batt_climb * t_climb
        energy_battery = energy_battery_hover + energy_battery_climb + energy_battery_cruise

        # GH2
        rH2_hover = (P_fc_hover) / (eta_fc * LHV_H2) # [kg/h]
        rH2_climb = (P_fc_climb) / (eta_fc * LHV_H2)
        rH2_cruise = (P_fc_cruise) / (eta_fc * LHV_H2)

        # Mass models (kg)
        mass_h2 = (rH2_hover * t_hover) + (rH2_cruise * t_cruise) + (rH2_climb * t_climb)
        mass_GH2 = mass_h2 / GI
        mass_FC = P_fc / rho_fc
        mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
        mass_motor_cruise = M_motor(P_out=P_cruise_motor_max, rho_motor=rho_motor_cruise)
        mass_motor_hover = M_motor(P_out=P_hover_motor_max, rho_motor=rho_motor_hover)
        mass_motor = mass_motor_cruise + mass_motor_hover
        mass_structures = MTOM * m_structures_factor

        mass_total = (
            mass_motor
            + m_payload
            + mass_battery
            + mass_GH2
            + mass_FC
            + m_flight_controller
            + m_flight_computer
            + m_servos
            + mass_auxilliary
            + mass_structures
        )

        # GH2 Tank Volume model
        volume_tank_m3 = V_hydrogen(Z=Z_tank, P_tank=P_tank, m_H2_stored=mass_h2, T=T_tank) # [m^3]
        volume_tank_L = volume_tank_m3 * 1000
        tank_l = ((volume_tank_m3 * (2*tank_AR)**2)/math.pi)**(1/3)
        tank_d = tank_l/tank_AR 

        # Fuselage Geometry model
        w_fuselage = 1.2 * tank_d
        l_fuselage = 2.2 * tank_l

        # Total onboard stored energy model. Battery mission energy is usable
        # electrical energy, so divide by usable fraction to get stored pack
        # capacity. Hydrogen is counted by stored chemical LHV.
        energy_battery_stored = energy_battery / max(e_usable, 1e-12)
        energy_h2_stored = mass_h2 * LHV_H2
        energy_stored_onboard = energy_battery_stored + energy_h2_stored

        # Constraints (<= 0 feasible)
        G = np.array(
            [
                MTOM - MTOM_max,
                V_min_cruise - V_cruise,
                V_min_climb - V_climb,
                stall_margin - (V_cruise - V_stall),
                b_wing - b_wing_max,
                abs(MTOM - mass_total) - tol_mass_kg,
                w_fuselage - 2*R_cruise,
                (2*R_hover + 0.03 + R_cruise) - 0.5*b_wing,
                (2*1.2*R_hover + 2*c + 2*R_hover) - l_max,
            ],
            dtype=float,
        )

        # Objectives
        F = np.array([energy_stored_onboard, t_trip], dtype=float)

        # Numerical guards
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
            "P_fc": P_fc,
            "t_cruise": t_cruise,
            "t_climb": t_climb,
            "t_trip": t_trip,
            "energy_stored_onboard": energy_stored_onboard,
            "energy_battery": energy_battery,
            "energy_battery_stored": energy_battery_stored,
            "energy_h2_stored": energy_h2_stored,
            "P_hover_motor": P_hover_motor_max,
            "P_cruise_motor": P_cruise_motor_max,
            "mass_battery": mass_battery,
            "mass_payload": m_payload,
            "mass_h2": mass_h2,
            "volume_tank_L": volume_tank_L,
            "tank_l": tank_l,
            "fuselage_l": l_fuselage,
            "fuselage_w": w_fuselage,
            "max_length": l_max,
            "tank_d": tank_d,
            "mass_GH2": mass_GH2,
            "mass_FC": mass_FC,
            "mass_structures": mass_structures,
            "mass_servos": m_servos,
            "mass_flight_controller": m_flight_controller,
            "mass_flight_computer": m_flight_computer,
            "mass_auxilliary": mass_auxilliary,
            "hover_motor_mass_per_rotor": (mass_motor_hover / num_rotors_hover) if num_rotors_hover > 0 else float("inf"),
            "cruise_motor_mass_per_rotor": (mass_motor_cruise / num_rotors_cruise) if num_rotors_cruise > 0 else float("inf"),
        }

        return F, G, debug

    
    # --- 3) pymoo Problem wrapper ---
    class EVTOLUAVTripProblem(Problem):
        def __init__(self):
            xl = np.array(
                [
                    MTOM_lb,     # MTOM [kg]
                    S_lb,        # S [m^2]
                    V_cruise_lb, # V_cruise [m/s]
                    V_climb_lb,  # V_climb [m/s]
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
                    V_climb_ub,  # V_climb [m/s]
                    AR_ub,       # AR [-]
                    R_cruise_ub, # R_cruise [m]
                    R_hover_ub,  # R_hover  [m]
                ],
                dtype=float,
            )
            super().__init__(n_var=7, n_obj=2, n_ieq_constr=9, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 2), dtype=float)
            G = np.zeros((X.shape[0], 9), dtype=float)
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
            "status": "no_pareto_set_found",
            "message": "NSGA-II did not return a Pareto set (res.X/res.F is None). Try relaxing constraints or increasing pop/gen.",
        }

    X_pareto = np.atleast_2d(res.X)
    F_pareto = np.atleast_2d(res.F)  # [:,0]=energy_stored_onboard [Wh], [:,1]=t_trip [h]

    # --- 5) Choose compromise point ---
    # Normalize robustly (avoid divide-by-zero)
    E = F_pareto[:, 0]
    T = F_pareto[:, 1]
    E_min = float(np.min(E)) if np.all(np.isfinite(E)) else 1.0
    T_min = float(np.min(T)) if np.all(np.isfinite(T)) else 1.0
    E_min = E_min if abs(E_min) > 1e-12 else 1.0
    T_min = T_min if abs(T_min) > 1e-12 else 1.0

    E_norm = E / E_min
    T_norm = T / T_min
    score = wE * E_norm + wT * T_norm
    best_idx = int(np.argmin(score))

    x_best = X_pareto[best_idx]
    f_best = F_pareto[best_idx]
    _, g_best, dbg_best = evaluate_trip(x_best)

    pareto_plot_file = None
    if plot_pareto:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.scatter(F_pareto[:, 0] / 1000.0, F_pareto[:, 1] * 60.0, s=28, alpha=0.85, label="Pareto set")
        ax.scatter(f_best[0] / 1000.0, f_best[1] * 60.0, s=72, marker="*", color="tab:red", label="Selected compromise")
        ax.set_xlabel("Onboard stored energy [kWh]")
        ax.set_ylabel("Total trip time [min]")
        ax.set_title("Pareto Front: Onboard Stored Energy vs Trip Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if pareto_plot_path:
            fig.savefig(pareto_plot_path, dpi=200, bbox_inches="tight")
            pareto_plot_file = str(pareto_plot_path)
        else:
            plt.show(block=pareto_plot_block)

    
    # --- 6) Assemble outputs ---
    output_dict = {}
    output_dict["MTOM"] = dbg_best["MTOM"]                      # [kg]
    output_dict["S_wing"] = dbg_best["S"]                       # [m^2]
    output_dict["AR_wing"] = dbg_best["AR"]                     # [-]
    output_dict["b_wing"] = dbg_best["b_wing"]                  # [m]
    output_dict["V_cruise"] = dbg_best["V_cruise"]              # [m/s]
    output_dict["V_climb"] = dbg_best["V_climb"]                # [m/s]
    output_dict["V_stall"] = dbg_best["V_stall"]                # [m/s]
    output_dict["LD_cruise"] = dbg_best["LD_cruise"]        # [-]
    output_dict["radius_cruise_propeller"] = dbg_best["R_cruise"]  # [m]
    output_dict["radius_hover_propeller"] = dbg_best["R_hover"]    # [m]

    # Performance
    output_dict["t_trip"] = dbg_best["t_trip"] * 60.0       # [min]
    output_dict["t_cruise"] = dbg_best["t_cruise"] * 60.0   # [min]
    output_dict["t_climb"] = dbg_best["t_climb"] * 60.0     # [min]
    output_dict["e_trip"] = dbg_best["energy_stored_onboard"]     # [Wh]
    output_dict["energy_stored_onboard"] = dbg_best["energy_stored_onboard"]     # [Wh]
    output_dict["energy_battery"] = dbg_best["energy_battery"]     # [Wh]
    output_dict["energy_battery_stored"] = dbg_best["energy_battery_stored"]     # [Wh]
    output_dict["energy_h2_stored"] = dbg_best["energy_h2_stored"]     # [Wh]

    # Aero at cruise
    output_dict["CL_cruise"] = dbg_best["CL_cruise"]
    output_dict["CD_cruise"] = dbg_best["CD_cruise"]

    # Powers
    output_dict["P_hover"] = dbg_best["P_hover"]
    output_dict["P_cruise"] = dbg_best["P_cruise"]
    output_dict["P_climb"] = dbg_best["P_climb"]
    output_dict["P_motor_hover_bldc"] = dbg_best["P_hover_motor"] / num_rotors_hover
    output_dict["P_motor_cruise_bldc"] = dbg_best["P_cruise_motor"] / num_rotors_cruise
    output_dict["P_fc"] = dbg_best["P_fc"]

    # Mass breakdown
    output_dict["mass_battery"] = dbg_best["mass_battery"]
    output_dict["mass_payload"] = dbg_best["mass_payload"]
    output_dict["mass_hover_bldc_per_rotor"] = dbg_best["hover_motor_mass_per_rotor"]
    output_dict["mass_cruise_bldc_per_rotor"] = dbg_best["cruise_motor_mass_per_rotor"]
    output_dict["mass_fuel_cell"] = dbg_best["mass_FC"]
    output_dict["mass_hydrogen_fuel"] = dbg_best["mass_h2"]
    output_dict["mass_storage_system"] = dbg_best["mass_GH2"]
    output_dict["mass_structures"] = dbg_best["mass_structures"]
    output_dict["mass_servos"] = dbg_best["mass_servos"]
    output_dict["mass_flight_controller"] = dbg_best["mass_flight_controller"]
    output_dict["mass_flight_computer"] = dbg_best["mass_flight_computer"]
    output_dict["mass_auxilliary"] = dbg_best["mass_auxilliary"]
    output_dict["volume_tank_L"] = dbg_best["volume_tank_L"]
    output_dict["tank_l"] = dbg_best["tank_l"]
    output_dict["tank_d"] = dbg_best["tank_d"]
    output_dict["fuselage_l"] = dbg_best["fuselage_l"]
    output_dict["fuselage_w"] = dbg_best["fuselage_w"]
    output_dict["max_length"] = dbg_best["max_length"]
    
    # Derived
    output_dict["wing_loading"] = (dbg_best["MTOM"] * g) / max(dbg_best["S"], 1e-12)
    output_dict["disk_loading"] = (dbg_best["MTOM"] * g) / (
        max(num_rotors_hover, 1) * math.pi * max(dbg_best["R_hover"], 1e-12) ** 2
    )

    # Optimization diagnostics
    output_dict["status"] = "ok"
    output_dict["best_idx"] = best_idx
    output_dict["best_objectives"] = {"energy_stored_onboard_Wh": float(f_best[0]), "t_trip_h": float(f_best[1])}
    output_dict["max_constraint_violation"] = float(np.max(np.maximum(0.0, g_best)))
    output_dict["pareto_set_size"] = int(len(X_pareto))
    output_dict["pareto_X"] = X_pareto
    output_dict["pareto_F"] = F_pareto
    output_dict["pareto_plot_path"] = pareto_plot_file

    return output_dict


input_dict = {
    # Mission
    "t_hover": 10.0,                   # [min] total hover time
    "range_cruise": 700,               # [km] cruise range

    # Geometric
    "l_max": 2.0,                      # [m] maximum length of UAV

    # Aerodynamics
    "CD0": 0.025,
    "CL_max": 1.6,
    "stall_margin": 5.0,               # [m/s]
    "e_wing": 0.85, 

    # Battery
    "rho_battery": 180.0,              # [Wh/kg]
    "e_usable_battery": 0.85,          # [-]

    # Fuel cell
    "eta_fc": 0.50,                    # [-]
    "rho_fc": 800.0,                   # [W/kg] specific power of FC system

    # GH2 storage
    "gravimetric_index": 0.054,         # [-] hydrogen mass / total tank-system mass
    "LHV_H2": 33330.0,                 # [Wh/kg]
    "tank_AR": 3,                      # [-] gaseous H2 tank length to diameter ratio
    "Z_tank": 1.236,                   # [-] compressibility factor at 35 MPa and 20 degC
    "P_tank": 35e6,                    # [Pa] tank pressure
    "T_tank": 293,                     # [K] tank temperature 

    # Hover propulsion
    "eta_elec_hover_bldc": 0.95,       # [-]
    "rho_motor_hover_bldc": 3000.0,    # [W/kg]
    "num_rotors_hover_propeller": 4,   # [-]
    "eta_prop_hover_propeller": 0.75,  # [-]

    # Cruise propulsion
    "eta_elec_cruise_bldc": 0.95,       # [-]
    "rho_motor_cruise_bldc": 3000.0,    # [W/kg]
    "num_rotors_cruise_propeller": 1,   # [-]
    "eta_prop_cruise_propeller": 0.85,  # [-]

    # Flight condition
    "b_cruise": 1.0,                   # [-] mass fraction used in P_cruise calculation
    "b_climb": 1.0,                    # [-] mass fraction used in P_climb calculation
    "theta_climb": 10,                 # [deg] maximum climb angle in fixed wing flight
    "hybrid_batt_factor_hover": 0.10,  # [-] fraction of hover power supplied by battery during hover
    "hybrid_batt_factor_cruise": 0.05, # [-] fraction of cruise power supplied by battery during cruise
    "hybrid_batt_factor_climb": 0.10,  # [-] fraction of climb power supplied by battery during climb
    "h_cruise": 5000.0,                # [m]
    "h_hover": 100.0,                  # [m]
    "h_climb_start": 100.0,            # [m]
    "g": 9.81,                         # [m/s^2]
    "vertical_climb_factor": 1.0,      # [-]

    # Fixed masses
    "mass_payload": 3,                 # [kg] mass of payload (EO/IR gimbal camera system)
    "mass_auxilliary": 0.5,            # [kg] mass of auxilliary electronics, avionics, cabling, etc. 
    "mass_flight_controller": 0.073,   # [kg] mass of pixhawk cube orange 
    "mass_flight_computer": 0.080,     # [kg] mass of RPi 5
    "num_servo": 4,                    # [-] number of servos
    "mass_servo": 0.06,                # [kg]
    "mass_structures_factor": 0.35,    # [-] mass of structures over mtom

    # Design variable bounds
    "MTOM_bounds": [1.0, 25.0],                        # [kg]
    "V_cruise_bounds": [15.0, 28],                  # [m/s]
    "V_climb_bounds": [15.0, 28],                  # [m/s]
    "S_wing_bounds": [0.35, 20],                    # [m^2]
    "AR_wing_bounds": [6.0, 10.0],                    # [-]
    "radius_cruise_propeller_bounds": [2*0.0254, 12*0.0254],  # [m]
    "radius_hover_propeller_bounds": [2*0.0254, 12*0.0254],   # [m]

    # Constraints
    "b_wing_bounds": [0.0, 3.0],       # [m]
    "tol_mass_kg": 0.05                # [kg]
}

# Example run
result = small_UAV_eVTOL_MOO(input_dict)

print("Status:", result["status"])
print("Best objective:", result["best_objectives"])
print("")
print("AIRCRAFT PARAMETERS")
print("MTOM [kg]:", result["MTOM"])
print("Wing area [m^2]:", result["S_wing"])
print("AR_wing [-]", result["AR_wing"])
print("b_wing [m]", result["b_wing"])
print("V_cruise [m/s]", result["V_cruise"])
print("V_climb [m/s]", result["V_climb"])
print("V_stall [m/s]", result["V_stall"])
print("L_over_D_cruise [-]", result["LD_cruise"])
print("radius_cruise_propeller [cm]", result["radius_cruise_propeller"]*100)
print("radius_hover_propeller [cm]", result["radius_hover_propeller"]*100)
print("")
print("MASS BREAKDOWN")
print("mass_battery [kg]", result["mass_battery"])
print("mass_payload [kg]", result["mass_payload"])
print("mass_GH2 [kg]", result["mass_storage_system"])
print("mass_FC [kg]", result["mass_fuel_cell"])
print("mass_h2 [kg]", result["mass_hydrogen_fuel"])
print("mass_flight_controller [kg]", result["mass_flight_controller"])
print("mass_flight_computer [kg]", result["mass_flight_computer"])
print("mass_auxilliary [kg]", result["mass_auxilliary"])
print("mass_structures [kg]", result["mass_structures"])
print("mass_servos [kg]", result["mass_servos"])
print("mass_hover_bldc_per_rotor [kg]", result["mass_hover_bldc_per_rotor"])
print("mass_cruise_bldc_per_rotor [kg]", result["mass_cruise_bldc_per_rotor"])
print("")
print("DIMENSIONS")
print("GH2_tank_volume [L]", result["volume_tank_L"])
print("GH2_tank_length [cm]", result["tank_l"]*100)
print("GH2_tank_diameter [cm]", result["tank_d"]*100)
print("fuselage_width [cm]", result["fuselage_w"]*100)
print("fuselage_length [cm]", result["fuselage_l"]*100)
print("max_x_length [cm]", result["max_length"]*100)
print("")
print("POWER BREAKDOWN")
print("Hover power [W]", result["P_hover"])
print("Cruise power [W]", result["P_cruise"])
print("Climb power [W]", result["P_climb"])
print("Fuel cell maximum power [W]", result["P_fc"])
print("Hover motor maximum power [W]", result["P_motor_hover_bldc"])
print("Cruise motor maximum power [W]", result["P_motor_cruise_bldc"])
print("")
print("ENERGY BREAKDOWN")
print("Battery energy [Wh]", result["energy_battery"])
