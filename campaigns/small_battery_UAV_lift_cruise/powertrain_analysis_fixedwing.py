""" 
This script computes the power the energy storage system, ESCs, and motors must produce 
to sustain steady-level cruise and constant speed climb. 

References:
[1] Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs.
[2] AER1216 Lecture Notes
[3] Data for 16 x 10 propeller gathered from: https://m-selig.ae.illinois.edu/props/volume-4/propDB-volume-4.html

"""


import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import math


# --- 1) Define Component Parameters ---
# Propeller parameters 
D = 9 * 0.0254 # propeller diameter [m]

data_4004_rpm = {'J': [0.0,
       0.151,
       0.191,
       0.236,
       0.276,
       0.325,
       0.361,
       0.409,
       0.444,
       0.486,
       0.524,
       0.573,
       0.61,
       0.649],
 'CT': [0.09104000000000002,
        0.0699,
        0.0643,
        0.058,
        0.052,
        0.046,
        0.0408,
        0.0323,
        0.0265,
        0.0194,
        0.0132,
        0.0041,
        -0.0019,
        -0.0099],
 'CP': [0.040175,
        0.0364,
        0.0354,
        0.0338,
        0.032,
        0.0304,
        0.0286,
        0.0255,
        0.0229,
        0.0197,
        0.0169,
        0.013,
        0.01,
        0.0066],
 'eta': [0.0,
         0.29,
         0.347,
         0.404,
         0.448,
         0.493,
         0.513,
         0.518,
         0.513,
         0.478,
         0.409,
         0.181,
         -0.118,
         -0.979]}

data_5007_rpm = {'J': [0.0,
       0.121,
       0.152,
       0.187,
       0.218,
       0.25,
       0.287,
       0.32,
       0.35,
       0.39,
       0.415,
       0.451,
       0.486,
       0.515,
       0.545,
       0.576,
       0.613,
       0.643],
 'CT': [0.09587419354838711,
        0.0787,
        0.0743,
        0.0691,
        0.0645,
        0.0586,
        0.053,
        0.0479,
        0.0434,
        0.0374,
        0.0332,
        0.027,
        0.0209,
        0.0154,
        0.0097,
        0.004,
        -0.0027,
        -0.0079],
 'CP': [0.03935161290322581,
        0.0374,
        0.0369,
        0.0362,
        0.0354,
        0.0342,
        0.0327,
        0.0312,
        0.0297,
        0.0276,
        0.0258,
        0.0233,
        0.0209,
        0.0185,
        0.0161,
        0.0133,
        0.01,
        0.0074],
 'eta': [0.0,
         0.256,
         0.306,
         0.357,
         0.397,
         0.429,
         0.466,
         0.491,
         0.51,
         0.528,
         0.533,
         0.523,
         0.486,
         0.427,
         0.328,
         0.171,
         -0.167,
         -0.685]}

data_6003_rpm = {'J': [0.0,
       0.101,
       0.128,
       0.154,
       0.182,
       0.211,
       0.238,
       0.265,
       0.292,
       0.318,
       0.348,
       0.372,
       0.403,
       0.426,
       0.454,
       0.476,
       0.507,
       0.535],
 'CT': [0.09632222222222218,
        0.0851,
        0.0821,
        0.0786,
        0.0744,
        0.0701,
        0.0658,
        0.0613,
        0.0564,
        0.0518,
        0.0462,
        0.0424,
        0.037,
        0.0332,
        0.0276,
        0.0243,
        0.0188,
        0.0134],
 'CP': [0.037125925925925916,
        0.0375,
        0.0376,
        0.0374,
        0.0369,
        0.0363,
        0.0355,
        0.0345,
        0.0334,
        0.0321,
        0.0304,
        0.0291,
        0.0272,
        0.0257,
        0.0234,
        0.0222,
        0.02,
        0.0178],
 'eta': [0.0,
         0.228,
         0.28,
         0.324,
         0.366,
         0.407,
         0.441,
         0.47,
         0.494,
         0.514,
         0.529,
         0.542,
         0.548,
         0.551,
         0.535,
         0.52,
         0.478,
         0.404]}

data_6811_rpm = {'J': [0.0,
       0.086,
       0.108,
       0.133,
       0.16,
       0.181,
       0.209,
       0.231,
       0.255,
       0.281,
       0.304,
       0.329,
       0.351,
       0.378,
       0.398,
       0.426,
       0.447,
       0.475],
 'CT': [0.09919090909090908,
        0.0902,
        0.0879,
        0.0853,
        0.082,
        0.0789,
        0.0745,
        0.0709,
        0.0667,
        0.0617,
        0.0576,
        0.053,
        0.049,
        0.0436,
        0.0402,
        0.0352,
        0.0311,
        0.0264],
 'CP': [0.0375,
        0.0375,
        0.0375,
        0.0376,
        0.0376,
        0.0375,
        0.037,
        0.0364,
        0.0357,
        0.0346,
        0.0336,
        0.0323,
        0.0312,
        0.0294,
        0.0282,
        0.0263,
        0.0248,
        0.023],
 'eta': [0.0,
         0.207,
         0.254,
         0.302,
         0.348,
         0.381,
         0.421,
         0.45,
         0.477,
         0.502,
         0.522,
         0.539,
         0.551,
         0.56,
         0.569,
         0.57,
         0.561,
         0.546]}

data_6815_rpm = {'J': [0.0,
       0.401,
       0.425,
       0.444,
       0.474,
       0.493,
       0.519,
       0.543,
       0.569,
       0.589,
       0.612,
       0.641],
 'CT': [0.11311666666666681,
        0.0396,
        0.0352,
        0.0317,
        0.0266,
        0.0232,
        0.0181,
        0.0138,
        0.0088,
        0.005,
        0.0006,
        -0.0054],
 'CP': [0.05463333333333338,
        0.0279,
        0.0263,
        0.025,
        0.023,
        0.0217,
        0.0196,
        0.0177,
        0.0154,
        0.0136,
        0.0114,
        0.0083],
 'eta': [0.0,
         0.568,
         0.568,
         0.563,
         0.548,
         0.527,
         0.478,
         0.424,
         0.326,
         0.217,
         0.033,
         -0.418]}

n_max = 19000/60

# ESC parameters
r_esc_on = 2.2e-3# esc on resistance [Ohm]
rise_time = 5.7e-9 # rise time [sec]
fall_time = 11.1e-9 # fall time [sec]
f_sw = 16e3 # switching frequncy in Hz
P_ic = 0.5 # power consumption of the ESC IC [W]
num_switches_effective = 6.0

# BLDC Parameters
Kv = 1115 * 2 * math.pi/60 # motor velocity constant [rad/s/V]
Kt = 1/Kv # motor torque constant
i0 = 1.8 # motor no load current [A]
rm = 0.011 # motor internal resistance [Ohms]

# LiPo Parameters
num_cells = 6
cell_voltage = 3.8 # nominal cell voltage of a LiPo

# UAV parameters 
MTOM = 3.0 # total mass [kg]
g = 9.81 # acceleration due to gravity [m*s^-2]
h_cruise = 90 # [m]
h_climb_start = 10 # [m]
rho = float(np.squeeze(Atmosphere(h_cruise).density)) # air density at cuise altitude [kg/m^3]
num_rotors = 1 # number of rotors for hover [-]
V_climb = 15 # maximum climb speed [m/s]
V_cruise = 18 # maximum cruise speed [m/s]
V_stall = 10 # stall speed [m/s]
CD0 = 0.03 # parasitic drag coefficient
AR = 8.5 # wing aspect ratio
CL_cruise = 0.5 # lift coefficient at cruise of 3D wing
CL_max = 1.6 # maximum lift coefficient of 3D wing
S_cruise = (MTOM*g)/(0.5*rho*CL_cruise*V_cruise**2)
S_stall = (MTOM*g)/(0.5*rho*CL_max*V_stall**2)
S_ref = max(S_cruise, S_stall)
e = 0.85 # 3D wing Oswald's efficiency 
theta_climb = 10 # climb angle in degreees

# Mission Parameters
t_cruise = 0.5 # hours 
t_climb = ((h_cruise - h_climb_start) / ((math.sin(math.radians(theta_climb))) * V_climb)) / 3600 # hours


# --- 2) Helper Functions
def get_coeff_at_J_interp(propeller_data, target_J, coeff_name):

    """ Return propeller coefficient from linear interpolation at target advance ratio."""

    J_values = np.asarray(propeller_data["J"], dtype=float)
    CT_values = np.asarray(propeller_data["CT"], dtype=float)
    CP_values = np.asarray(propeller_data["CP"], dtype=float)
    eta_values = np.asarray(propeller_data["eta"], dtype=float)

    sorted_indices = np.argsort(J_values)
    J_values = J_values[sorted_indices]
    CT_values = CT_values[sorted_indices]
    CP_values = CP_values[sorted_indices]
    eta_values = eta_values[sorted_indices]

    if coeff_name == "CT":
        return float(np.interp(target_J, J_values, CT_values))
    elif coeff_name == "CP":
        return float(np.interp(target_J, J_values, CP_values))
    elif coeff_name == "eta":
        return float(np.interp(target_J, J_values, eta_values))
    else:
        print("Coefficient does not exist")


def select_nearest_propeller_dataset(n_rev_s):
    """Select the measured propeller dataset closest to the operating RPM."""
    rpm = 60.0 * n_rev_s
    nearest_rpm = min(propeller_datasets.keys(), key=lambda rpm_key: abs(rpm_key - rpm))
    return propeller_datasets[nearest_rpm], nearest_rpm


def propeller_performance_per_rotor(n_rev_s, V, rho):
    """
    Per-rotor propeller performance at a given rotor speed and axial speed.
    n_rev_s is in rev/s, not RPM.

    """
    if n_rev_s <= 0:
        raise ValueError("Rotor speed must be positive.")

    prop_data, dataset_rpm = select_nearest_propeller_dataset(n_rev_s)
    J = V / (n_rev_s * D)

    CT = get_coeff_at_J_interp(prop_data, J, "CT")
    CP = get_coeff_at_J_interp(prop_data, J, "CP")

    T = CT * rho * n_rev_s**2 * D**4
    P_shaft = CP * rho * n_rev_s**3 * D**5
    omega = 2.0 * math.pi * n_rev_s
    Q_shaft = P_shaft / omega

    return {
        "n_rev_s": n_rev_s,
        "RPM": 60.0 * n_rev_s,
        "dataset_RPM": dataset_rpm,
        "J": J,
        "CT": CT,
        "CP": CP,
        "T_per_rotor_N": T,
        "P_shaft_per_rotor_W": P_shaft,
        "Q_shaft_per_rotor_Nm": Q_shaft,
    }


def aircraft_performance(n_rev_s, V, rho):

    """
    Total aircraft propulsive performance from per-rotor performance.

    """
    per = propeller_performance_per_rotor(n_rev_s, V, rho)
    total = dict(per)
    total["T_total_N"] = num_rotors * per["T_per_rotor_N"]
    total["P_shaft_total_W"] = num_rotors * per["P_shaft_per_rotor_W"]
    return total


def required_total_thrust(V, rho, theta_climb):

    """
    Steady cruise and climb thrust required.
    
    """

    K = 1.0 / (math.pi * e * AR)
    CL = (2.0 * MTOM * g * math.cos(math.radians(theta_climb))) / (rho * S_ref * V ** 2)
    CD = CD0 + K * CL**2
    drag = 0.5 * rho * S_ref * CD * V**2
    weight = MTOM * g
    thrust = drag + weight * math.sin(math.radians(theta_climb))
    return thrust


def solve_rotor_speed_for_thrust(V, rho, theta_climb, n_low=1.0, n_high=n_max):
    """Bisection solve for rotor speed that gives required steady-climb thrust."""
    target_T = required_total_thrust(V, rho, theta_climb)

    T_low = aircraft_performance(n_low, V, rho)["T_total_N"]
    T_high = aircraft_performance(n_high, V, rho)["T_total_N"]

    if T_high < target_T:
        raise RuntimeError(
            f"n_max={n_high:.2f} rev/s cannot generate enough thrust. "
            f"T_max={T_high:.2f} N, required={target_T:.2f} N."
        )

    if T_low > target_T:
        raise RuntimeError(
            f"n_low={n_low:.2f} rev/s already exceeds required thrust. "
            "Reduce n_low."
        )

    for _ in range(80):
        n_mid = 0.5 * (n_low + n_high)
        T_mid = aircraft_performance(n_mid, V, rho)["T_total_N"]

        if T_mid < target_T:
            n_low = n_mid
        else:
            n_high = n_mid

    n_solution = 0.5 * (n_low + n_high)
    perf = aircraft_performance(n_solution, V, rho)
    perf["T_required_total_N"] = target_T
    perf["body_drag_N"] = target_T - MTOM * g * math.sin(math.radians(theta_climb))
    return perf

# Motor, ESC, battery models
def motor_from_shaft_power(P_shaft_per_rotor_W, n_rev_s):
    """Approximate BLDC electrical state for one motor from per-rotor shaft power."""
    omega = 2.0 * math.pi * n_rev_s
    Q = P_shaft_per_rotor_W / omega

    I_motor = Q / Kt + i0
    V_motor = omega / Kv + I_motor * rm
    P_motor_elec = V_motor * I_motor
    eta_motor = P_shaft_per_rotor_W / P_motor_elec if P_motor_elec > 0 else np.nan

    return {
        "omega_rad_s": omega,
        "Q_shaft_Nm": Q,
        "P_shaft_W": P_shaft_per_rotor_W,
        "I_motor_A": I_motor,
        "V_motor_V": V_motor,
        "P_motor_electrical_W": P_motor_elec,
        "eta_motor": eta_motor,
    }


def esc_battery_hydrogen_from_motor(motor, duty, mission_duration):

    """Approximate per-ESC and total battery and hydrogen quantities from 
    one motor state. Battery and fuel cell are connected in parallel. """

    if duty <= 0 or duty > 1:
        raise ValueError("duty must be in (0, 1].")

    V_bus_ideal = motor["V_motor_V"] / duty
    I_motor = motor["I_motor_A"]

    P_cond = duty * I_motor**2 * r_esc_on
    P_sw = 0.5 * V_bus_ideal * I_motor * (rise_time + fall_time) * f_sw * num_switches_effective
    P_esc_loss = P_cond + P_sw + P_ic

    P_bus_per_rotor = motor["P_motor_electrical_W"] + P_esc_loss
    I_bus_per_rotor = P_bus_per_rotor / V_bus_ideal
    P_bus_total_W = num_rotors * P_bus_per_rotor

    P_batt_required = P_bus_total_W

    C_batt = (P_batt_required * mission_duration) / (V_bus_ideal)

    return {
        "V_bus_V": V_bus_ideal,
        "P_esc_loss_per_rotor_W": P_esc_loss,
        "P_bus_per_rotor_W": P_bus_per_rotor,
        "I_bus_per_rotor_A": I_bus_per_rotor,
        "P_bus_total_W": num_rotors * P_bus_per_rotor,
        "I_bus_total_A": num_rotors * I_bus_per_rotor,
        "P_batt_required_W": P_batt_required,
        "C_batt_required_Ah": C_batt
    }


# --- 3) Data Visualization ---
propeller_datasets = {
    4004.0: data_4004_rpm,
    5007.0: data_5007_rpm,
    6003.0: data_6003_rpm,
    6811.0: data_6811_rpm,
    6815.0: data_6815_rpm,
}


fit_variables = {
    "CT": "Thrust coefficient, $C_T$",
    "CP": "Power coefficient, $C_P$",
    "eta": "Efficiency, $\\eta$",
}

fig, axes = plt.subplots(len(fit_variables), 1, sharex=True, figsize=(9, 10))

for axis, (variable_name, variable_label) in zip(axes, fit_variables.items()):
    for rpm_label, propeller_data in propeller_datasets.items():
        J_values = np.asarray(propeller_data["J"], dtype=float)
        variable_values = np.asarray(propeller_data[variable_name], dtype=float)
        sorted_indices = np.argsort(J_values)
        J_values = J_values[sorted_indices]
        variable_values = variable_values[sorted_indices]
        J_interp = np.linspace(J_values.min(), J_values.max(), 200)
        variable_interp = np.interp(J_interp, J_values, variable_values)

        data_plot, = axis.plot(J_values, variable_values, "x", label=f"{rpm_label} data")
        axis.plot(
            J_interp,
            variable_interp,
            "-",
            color=data_plot.get_color(),
            label=f"{rpm_label} interp",
        )

    axis.set_ylabel(variable_label)
    axis.grid(True)

axes[-1].set_xlabel("Advance ratio, J")
fig.suptitle("Propeller Data with Linear Interpolation")
fig.tight_layout()
fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.show()


# --- 4) Run Performance Analysis ---
def full_powertrain_result(V, theta_climb, duty, mission_duration):
    prop = solve_rotor_speed_for_thrust(V, rho, theta_climb)
    motor = motor_from_shaft_power(prop["P_shaft_per_rotor_W"], prop["n_rev_s"])
    battery = esc_battery_hydrogen_from_motor(motor, duty, mission_duration)

    result = {}
    result.update(prop)
    result.update(motor)
    result.update(battery)
    result["V_forward_mps"] = V
    return result


def print_result(title, result):
    print(f"\n--- {title} ---")
    print(f"Wing reference area: {S_ref} m^2")
    print(f"Altitude: {h_cruise:.1f} m")
    print(f"Air density: {rho:.4f} kg/m^3")
    print(f"Forward speed: {result['V_forward_mps']:.3f} m/s")
    print(f"Rotor speed: {result['n_rev_s']:.3f} rev/s = {result['RPM']:.1f} RPM")
    print(f"Nearest prop dataset: {result['dataset_RPM']:.0f} RPM")
    print(f"J: {result['J']:.5f}, CT: {result['CT']:.5f}, CP: {result['CP']:.5f}")
    print(f"Total thrust required: {result['T_required_total_N']:.2f} N")
    print(f"Total thrust generated: {result['T_total_N']:.2f} N")
    print(f"Per-rotor shaft torque: {result['Q_shaft_per_rotor_Nm']:.4f} Nm")
    print(f"Per-rotor shaft power: {result['P_shaft_per_rotor_W']:.1f} W")
    print(f"Total shaft power: {result['P_shaft_total_W']:.1f} W")
    print(f"Motor current per motor: {result['I_motor_A']:.1f} A")
    print(f"Motor voltage: {result['V_motor_V']:.1f} V")
    print(f"Motor efficiency estimate: {100.0 * result['eta_motor']:.1f} %")
    print(f"Bus voltage estimate: {result['V_bus_V']:.1f} V")
    print(f"Bus current: {result['I_bus_total_A']:.1f} A")
    print(f"Bus power total: {result['P_bus_total_W']:.1f} W")
    print(f"Battery power required: {result['P_batt_required_W']:.1f} W"),
    print(f"Battery capacity required: {result['C_batt_required_Ah']:.1f} Ah")



def run_case(title, V, theta_climb, duty, mission_duration):
    try:
        result = full_powertrain_result(V=V, theta_climb=theta_climb, duty=duty, mission_duration=mission_duration)
    except RuntimeError as exc:
        print(f"\n--- {title} ---")
        print(f"No feasible rotor speed found: {exc}")
        print("Increase n_max, increase prop diameter/count, reduce drag/weight, or reduce the required speed/climb angle.")
        return None

    print_result(title, result)
    return result


cruise = run_case("Cruise Performance", V=V_cruise, theta_climb=0, duty=0.38, mission_duration=t_cruise)
climb = run_case("Climb Performance", V=V_climb, theta_climb=theta_climb, duty=0.48, mission_duration=t_climb)
