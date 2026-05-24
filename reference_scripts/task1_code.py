### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt

### --- Aircraft Data ---
# Aerodynamic Data
CL_0 = 0.4 # Lift coefficient of aircraft at zero alpha [-]
CL_alpha = 5.7 # Lift coefficient vs. alpha curve slope [1/rad]
CL_at_CD_min = 0.15 # Lift coefficient of aircraft at minimum drag coefficient [-]
CD_min = 0.022 # Parasitic drag coefficient [-]
K = 0.035 # Induced drag coefficient [-]
alpha_stall = 15 # Angle of attack at stall [deg]
S_ref = 20 # Wing reference area [m^2]
g = 9.81 # Acceleration due to gravity [m/s^2]

# Mass Data
MTOM = 2500 # Maximum take-off mass [kg]
MTOW = 2500 * g # Maximum take-off weight [N]
V_fuel_max = 1200 # Maximum fuel volume [L]
rho_fuel = 800 # Fuel density [kg/m^3]
fuel_remaining = 100 # Volume of fuel remaining at the end of the mission [L]

V_fuel_max_m3 = V_fuel_max * 1e-3 # Maximum fuel volume [m^3]
V_fuel_final_m3 = fuel_remaining * 1e-3 # Remaining fuel volume [m^3]

m_fuel_max = V_fuel_max_m3 * rho_fuel # Starting fuel mass [kg]
m_fuel_final = V_fuel_final_m3 * rho_fuel # Remaining fuel mass [kg]

m_initial = MTOM
m_empty = MTOM - m_fuel_max
m_final_target = MTOM - (m_fuel_max-m_fuel_final)

# Propulsion data
eta_prop = 0.8 # Propeller efficiency [-]
P_TO_W = 400 * 745.7 # Take-off power [W]
SFC_TO_SI = 0.28 / (3600 * 1000) # Specific fuel consumption at take off power [kg/W/s]
P_max_cont_W = 350 * 745.7# Maximum continuous power [W]
SFC_max_cont_SI = 0.25 / (3600 * 1000) # Specific fuel consumption at max continuous power [kg/W/s]
P_cruise_W = 280 * 745.7# Cruise power [W]
SFC_cruise_SI = 0.2 / (3600 * 1000) # Specific fuel consumption at cruise power [kg/W/s]

# Mission data
alt_0 = 20000 * 0.3048 # Cruise start altitude [m]
atm = Atmosphere(alt_0) 
rho_air = float(np.squeeze(atm.density)) # Air density at 20000 ft altitude [kg/m^3]

### --- Analytical Study ---
# Compute minimum power airspeed
V_mp = ((math.sqrt(2)/math.sqrt(3)) * 
math.sqrt((math.sqrt(4 * CL_at_CD_min**2 * K**2 + 3 * CD_min * K) * MTOW) / (CL_at_CD_min**2 * K * S_ref * rho_air + CD_min * S_ref * rho_air) + 
(CL_at_CD_min * K * MTOW) / (CL_at_CD_min**2 * K * S_ref * rho_air + CD_min * S_ref * rho_air)))

# Compute lift coefficient at V_mp
CL_mp = (2 * MTOW) / (rho_air * S_ref * V_mp**2)

# Compute alpha at CL_mp
alpha_deg_mp = ((CL_mp - CL_0) / CL_alpha) * (180/math.pi)

# Compute drag coefficient at V_mp
CD_mp = CD_min + K*(CL_mp - CL_at_CD_min)**2

# Compute max endurance at constant airspeed
E_max_constant_airspeed = (eta_prop / (SFC_cruise_SI * g)) * (CL_mp / (V_mp * CD_mp)) * math.log(MTOM/m_final_target)

# Compute (CL/CD)_max 
CL_md = math.sqrt(CL_at_CD_min**2 + CD_min/K)
CD_md = CD_min + K*(CL_md - CL_at_CD_min)**2
CL_over_CD_max = CL_md / CD_md
V_md = math.sqrt((2*MTOW)/(rho_air * S_ref * CL_md))

# Compute alpha at (CL/CD)_max
alpha_deg_md = ((CL_md - CL_0) / CL_alpha) * (180/math.pi)

# Compute max endurance at constant altitude
E_max_constant_altitude = -(eta_prop / (SFC_cruise_SI * g)) * CL_over_CD_max * math.sqrt(rho_air * S_ref * CL_md * 0.5) * (2/math.sqrt(MTOW) - 2/math.sqrt(m_final_target*g))

# Compute max range at constant airspeed
R_max_constant_airspeed = (eta_prop / (SFC_cruise_SI * g)) * CL_over_CD_max * math.log(MTOM/m_final_target)


### --- Numerical Study ---
# Define aerodynamic helper functions
def CL_from_alpha(alpha_deg: float) -> float:
    alpha_rad = math.radians(alpha_deg)
    CL = CL_0 + CL_alpha * alpha_rad
    return CL

def CD_from_CL(CL:float) -> float:
    CD = CD_min + K * (CL - CL_at_CD_min)**2
    return CD

# Define constant altitude cruise mission
def simulate_constant_altitude(alpha_deg, dt = 1.0):
    """
    Computes max range and endurance at constant altitude
    of 20000 ft for a given angle of attack. Assumes 
    steady-level flight (L = W and D = T) at all times.

    - alpha_deg: angle of attack [deg]
    - dt: timestep [s]

    Returns a dict with histories
    """

    # Initial conditions
    atm = Atmosphere(alt_0)
    rho = float(np.squeeze(atm.density))

    # Get aerodynamic coefficients
    CL = CL_from_alpha(alpha_deg)
    CD = CD_from_CL(CL)

    # Data histories
    t_hist = []
    alt_hist = []
    m_hist = []
    m_fuel_hist = []
    W_hist = []
    V_hist = []
    R_hist = []

    # Initialize endurance, range, and starting fuel mass
    t = 0.0
    R = 0.0
    m_fuel = m_fuel_max

    # Define simulation loop
    while m_fuel > m_fuel_final:
        m = m_empty + m_fuel
        W = m * g

        # Compute TAS
        V = math.sqrt(2.0 * W / (rho * S_ref * CL))

        # Compute fuel burn
        D = 0.5 * rho * V**2 * S_ref * CD
        P_shaft = D * V / eta_prop
        m_dot_fuel = SFC_cruise_SI * P_shaft

        if m_dot_fuel <= 0.0:
            break

        # Record current state
        t_hist.append(t)
        alt_hist.append(alt_0)
        m_hist.append(m)
        m_fuel_hist.append(m_fuel)
        W_hist.append(W)
        V_hist.append(V)
        R_hist.append(R)

        # Integrate 
        t += dt
        R += V * dt
        m_fuel -= m_dot_fuel * dt

    return {
    "t": np.array(t_hist),
    "alt": np.array(alt_hist),
    "m": np.array(m_hist),
    "m_fuel": np.array(m_fuel_hist),
    "W": np.array(W_hist),
    "V": np.array(V_hist),
    "R": np.array(R_hist),
    }

# Define constant TAS cruise mission
def simulate_constant_tas(alpha_deg, V_const, dt=1.0):
    """
    Computes max range and endurance at constant TAS
    with varying altitude for a given angle of attack and 
    TAS. Assumes steady-level flight (L = W and D = T) 
    at all times.

    - alpha_deg: angle of attack [deg]
    - V_const: TAS [m/s]
    - dt: timestep [s]

    Returns a dict with histories.
    """

    # Get aerodynamic coefficients
    CL = CL_from_alpha(alpha_deg)
    CD = CD_from_CL(CL)

    # Histories
    t_hist = []
    alt_hist = []
    m_hist = []
    m_fuel_hist = []
    W_hist = []
    V_hist = []
    R_hist = []

    # State variables
    t = 0.0
    R = 0.0
    m_fuel = m_fuel_max

    while m_fuel > m_fuel_final:
        m = m_empty + m_fuel
        W = m * g

        # Density required for level flight at fixed V and CL
        rho_target = 2.0 * W / (S_ref * CL * V_const**2)

        D = 0.5 * rho_target * V_const**2 * S_ref * CD
        P_shaft = D * V_const / eta_prop
        m_dot_fuel = SFC_cruise_SI * P_shaft

        if m_dot_fuel <= 0.0:
            break

        # Record current state
        t_hist.append(t)
        alt_hist.append(rho_target)
        m_hist.append(m)
        m_fuel_hist.append(m_fuel)
        W_hist.append(W)
        V_hist.append(V_const)
        R_hist.append(R)

        # Integrate forward
        R += V_const * dt
        m_fuel -= m_dot_fuel * dt
        t += dt

    return {
        "t": np.array(t_hist),
        "alt": np.array(alt_hist),
        "m": np.array(m_hist),
        "m_fuel": np.array(m_fuel_hist),
        "W": np.array(W_hist),
        "V": np.array(V_hist),
        "R": np.array(R_hist),
    }

if __name__ == "__main__":

    constant_TAS_max_endurance_results = simulate_constant_tas(alpha_deg=alpha_deg_mp, V_const=V_mp, dt=1.0)
    constant_TAS_max_range_results = simulate_constant_tas(alpha_deg=alpha_deg_md, V_const=V_md, dt=1.0)

    E_max_constant_TAS_numerical = constant_TAS_max_endurance_results["t"][-1]
    R_max_constant_TAS_numerical = constant_TAS_max_range_results["R"][-1]

    constant_altitude_max_endurance_results = simulate_constant_altitude(alpha_deg=alpha_deg_md, dt=1.0)
    constant_altitude_max_range_results = simulate_constant_altitude(alpha_deg=alpha_deg_md, dt=1.0)

    E_max_constant_altitude_numerical = constant_altitude_max_endurance_results["t"][-1]
    R_max_constant_altitude_numerical = constant_altitude_max_range_results["R"][-1]

    print("--- MAX ENDURANCE ---")
    print("At constant airspeed")
    print(f"Best angle of attack: {alpha_deg_mp} degrees")
    print(f"Maximum endurance analytical: {E_max_constant_airspeed/3600} hours")
    print(f"Maximum endurance numerical: {E_max_constant_TAS_numerical/3600} hours")
    print(" ")
    print("At constant altitude")
    print(f"Best angle of attack: {alpha_deg_md} degrees")
    print(f"Maximum endurance analytical: {E_max_constant_altitude/3600} hours")
    print(f"Maximum endurance numerical: {E_max_constant_altitude_numerical/3600} hours")
    print(" ")
    print("--- MAX RANGE ---")
    print("At constant airspeed")
    print(f"Best angle of attack: {alpha_deg_md} degrees")
    print(f"Maximum range analytical: {R_max_constant_airspeed/1000} km")
    print(f"Maximum range numerical: { R_max_constant_TAS_numerical/1000} km")
    print(" ")
    print("At constant altitude")
    print(f"Best angle of attack: {alpha_deg_md} degrees")
    print(f"Maximum range: {R_max_constant_airspeed/1000} km")
    print(f"Maximum range numerical: {R_max_constant_altitude_numerical/1000} km")
    print(" ")

    # Global plotting style
    plt.rcParams.update({
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    def plot_mission_results(title: str, results: dict) -> None:
        """Create a 2x2 subplot of key histories for a mission result dict."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        t = results["t"]

        axes[0, 0].plot(t/3600, results["m"])
        axes[0, 0].set_ylabel("Mass [kg]")
        axes[0, 0].set_xlabel("Time [hours]")
        axes[0, 0].set_title("Aircraft Mass vs. Time")

        axes[0, 1].plot(t/3600, results["R"]/1000)
        axes[0, 1].set_ylabel("Range [km]")
        axes[0, 1].set_xlabel("Time [hours]")
        axes[0, 1].set_title("Range vs. Time")

        axes[1, 0].plot(t/3600, results["V"])
        axes[1, 0].set_ylabel("TAS [m/s]")
        axes[1, 0].set_xlabel("Time [hours]")
        axes[1, 0].set_title("TAS vs. Time")

        axes[1, 1].plot(t/3600, results["alt"])
        axes[1, 1].set_ylabel("Air Denstiy [kg/m^3]")
        axes[1, 1].set_xlabel("Time [hours]")
        axes[1, 1].set_title("Air Density at Cruise Altitude vs. Time")

        fig.suptitle(title)
        fig.tight_layout()

    scenarios = [
        ("Constant TAS - Max Endurance", constant_TAS_max_endurance_results),
        ("Constant TAS - Max Range", constant_TAS_max_range_results),
        ("Constant Altitude - Max Endurance", constant_altitude_max_endurance_results),
        ("Constant Altitude - Max Range", constant_altitude_max_range_results),
    ]

    for title, results in scenarios:
        plot_mission_results(title, results)
    
    plt.show()

    
