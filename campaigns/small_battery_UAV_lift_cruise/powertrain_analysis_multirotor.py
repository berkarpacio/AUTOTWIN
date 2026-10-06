""" 
This script computes the power the energy storage system, ESCs, and motors must produce 
to sustain hover and constant speed vertical climb. 

References:
[1] Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs.
[2] AER1216 Lecture Notes
[3] Data for 7x6 propeller gathered from: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html

"""


import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import math


# --- 1) Define Component Parameters ---
# Propeller parameters 
D = 7 * 0.0254 # propeller diameter [m]

data_4015_rpm = {'J': [0.0,
       0.202,
       0.251,
       0.305,
       0.355,
       0.412,
       0.47,
       0.514,
       0.566,
       0.625,
       0.67,
       0.732,
       0.766,
       0.835,
       0.887,
       0.92,
       0.991,
       1.017],
 'CT': [0.08992040816326535,
        0.0825,
        0.0807,
        0.0785,
        0.0755,
        0.0711,
        0.0654,
        0.0611,
        0.0551,
        0.0477,
        0.0416,
        0.0333,
        0.0291,
        0.0201,
        0.0122,
        0.0055,
        -0.0049,
        -0.0086],
 'CP': [0.07329795918367352,
        0.07,
        0.0692,
        0.0681,
        0.0666,
        0.0649,
        0.0631,
        0.0611,
        0.0575,
        0.0534,
        0.0484,
        0.0433,
        0.041,
        0.033,
        0.0276,
        0.0235,
        0.0132,
        0.0129],
 'eta': [0.0,
         0.238,
         0.292,
         0.351,
         0.403,
         0.451,
         0.487,
         0.514,
         0.542,
         0.558,
         0.577,
         0.564,
         0.544,
         0.508,
         0.393,
         0.214,
         -0.391,
         -0.681]}

data_5024_rpm = {'J': [0.0,
       0.158,
       0.203,
       0.243,
       0.283,
       0.326,
       0.371,
       0.414,
       0.455,
       0.495,
       0.542,
       0.579,
       0.611,
       0.659,
       0.695,
       0.744,
       0.779,
       0.827],
 'CT': [0.09011555555555555,
        0.0852,
        0.0838,
        0.0828,
        0.082,
        0.0802,
        0.078,
        0.0741,
        0.0704,
        0.0661,
        0.0599,
        0.0553,
        0.0511,
        0.0444,
        0.0401,
        0.0325,
        0.0281,
        0.0201],
 'CP': [0.0722555555555555,
        0.0705,
        0.07,
        0.0686,
        0.0674,
        0.0669,
        0.0668,
        0.0654,
        0.065,
        0.0631,
        0.0599,
        0.0578,
        0.0552,
        0.0512,
        0.0487,
        0.0439,
        0.0399,
        0.0353],
 'eta': [0.0,
         0.191,
         0.243,
         0.294,
         0.345,
         0.391,
         0.433,
         0.47,
         0.493,
         0.518,
         0.542,
         0.554,
         0.565,
         0.572,
         0.572,
         0.55,
         0.548,
         0.47]}

data_6023_rpm = {'J': [0.0,
       0.584,
       0.62,
       0.656,
       0.688,
       0.721,
       0.752,
       0.796,
       0.822,
       0.849,
       0.891,
       0.93,
       0.952,
       0.979,
       1.04],
 'CT': [0.13576666666666656,
        0.0579,
        0.0531,
        0.0472,
        0.0425,
        0.0378,
        0.0327,
        0.0252,
        0.0209,
        0.017,
        0.0112,
        0.0049,
        0.0016,
        -0.0036,
        -0.0139],
 'CP': [0.09641111111111103,
        0.0591,
        0.0568,
        0.0533,
        0.0505,
        0.0474,
        0.0444,
        0.04,
        0.0369,
        0.0338,
        0.0293,
        0.0249,
        0.0216,
        0.0186,
        0.0109],
 'eta': [0.0,
         0.572,
         0.58,
         0.581,
         0.578,
         0.575,
         0.553,
         0.503,
         0.465,
         0.427,
         0.341,
         0.184,
         0.07,
         -0.192,
         -1.335]}

data_6024_rpm = {'J': [0.0,
       0.134,
       0.167,
       0.202,
       0.238,
       0.272,
       0.307,
       0.339,
       0.377,
       0.411,
       0.448,
       0.484,
       0.518,
       0.545,
       0.585,
       0.613,
       0.653,
       0.682],
 'CT': [0.0875,
        0.0875,
        0.0875,
        0.0861,
        0.0859,
        0.0859,
        0.085,
        0.0836,
        0.0814,
        0.0789,
        0.0753,
        0.0711,
        0.0667,
        0.0635,
        0.0576,
        0.0539,
        0.0476,
        0.0435],
 'CP': [0.07252424242424248,
        0.0709,
        0.0705,
        0.0685,
        0.0676,
        0.0677,
        0.0681,
        0.0675,
        0.0676,
        0.067,
        0.0662,
        0.0649,
        0.063,
        0.0615,
        0.0594,
        0.0574,
        0.0542,
        0.0515],
 'eta': [0.0,
         0.166,
         0.207,
         0.254,
         0.302,
         0.345,
         0.384,
         0.42,
         0.454,
         0.484,
         0.509,
         0.53,
         0.549,
         0.563,
         0.567,
         0.575,
         0.574,
         0.576]}

data_7020_rpm = {'J': [0.0,
       0.11,
       0.142,
       0.172,
       0.2,
       0.231,
       0.263,
       0.294,
       0.321,
       0.355,
       0.384,
       0.413,
       0.443,
       0.472,
       0.497,
       0.527,
       0.554,
       0.583],
 'CT': [0.09326250000000001,
        0.0912,
        0.0906,
        0.0911,
        0.0901,
        0.0903,
        0.0902,
        0.0895,
        0.0884,
        0.0867,
        0.0851,
        0.083,
        0.0801,
        0.0772,
        0.0743,
        0.0699,
        0.0662,
        0.0623],
 'CP': [0.07448125000000001,
        0.0707,
        0.0696,
        0.069,
        0.0677,
        0.067,
        0.0671,
        0.0673,
        0.0673,
        0.0676,
        0.0677,
        0.0675,
        0.0667,
        0.0661,
        0.0654,
        0.0636,
        0.0621,
        0.0606],
 'eta': [0.0,
         0.142,
         0.185,
         0.227,
         0.267,
         0.311,
         0.353,
         0.391,
         0.422,
         0.456,
         0.482,
         0.508,
         0.533,
         0.551,
         0.564,
         0.579,
         0.591,
         0.6]}

data_7021_rpm = {'J': [0.0,
       0.5,
       0.528,
       0.558,
       0.584,
       0.617,
       0.652,
       0.682,
       0.708,
       0.735,
       0.769,
       0.796,
       0.828,
       0.853,
       0.884,
       0.911,
       0.943,
       0.977,
       1.002,
       1.035],
 'CT': [0.14741428571428578,
        0.0742,
        0.0701,
        0.066,
        0.062,
        0.0566,
        0.0499,
        0.0465,
        0.0413,
        0.0369,
        0.0305,
        0.0267,
        0.0231,
        0.0183,
        0.0134,
        0.0093,
        0.0038,
        -0.0024,
        -0.0068,
        -0.0128],
 'CP': [0.08444285714285694,
        0.0648,
        0.0637,
        0.062,
        0.0601,
        0.0576,
        0.0547,
        0.0528,
        0.0496,
        0.0471,
        0.0433,
        0.0407,
        0.0379,
        0.0347,
        0.0316,
        0.0285,
        0.0243,
        0.02,
        0.0166,
        0.0121],
 'eta': [0.0,
         0.572,
         0.581,
         0.594,
         0.602,
         0.606,
         0.594,
         0.6,
         0.59,
         0.576,
         0.541,
         0.523,
         0.504,
         0.449,
         0.375,
         0.299,
         0.147,
         -0.119,
         -0.412,
         -1.101]}

n_max = 400

# ESC parameters
r_esc_on = 2.2e-3# esc on resistance [Ohm]
rise_time = 5.7e-9 # rise time [sec]
fall_time = 11.1e-9 # fall time [sec]
f_sw = 16e3 # switching frequncy in Hz
P_ic = 0.5 # power consumption of the ESC IC [W]
num_switches_effective = 6.0

# BLDC Parameters
Kv = 1300 * 2 * math.pi/60 # motor velocity constant [rad/s/V]
Kt = 1/Kv # motor torque constant
i0 = 0.95 # motor no load current [A]
rm = 0.076 # motor internal resistance [Ohms]

# LiPo Parameters
num_cells = 6
cell_voltage = 3.8 # nominal cell voltage of a LiPo

# UAV parameters 
MTOM = 3.0 # total mass [kg]
g = 9.81 # acceleration due to gravity [m*s^-2]
hover_altitude = 10 # [m]
rho_hover = float(np.squeeze(Atmosphere(hover_altitude).density)) # air density at hover altitude [kg/m^3]
num_rotors = 4 # number of rotors for hover [-]
V_climb = 1.5 # maximum climb speed [m/s]
CD_frame = 0.5 # body drag coefficient [-]
S_ref = 0.9 # reference area for drag computation [m^2]

# Mission Parameters
t_hover = 3 * (1/60) # hours 
t_vertical_climb = 1 * (1/60) # hours


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


def propeller_performance_per_rotor(n_rev_s, axial_speed_mps, rho):
    """
    Per-rotor propeller performance at a given rotor speed and axial climb speed.

    n_rev_s is in rev/s, not RPM.
    axial_speed_mps is positive for vertical climb along the rotor axis.
    """
    if n_rev_s <= 0:
        raise ValueError("Rotor speed must be positive.")

    prop_data, dataset_rpm = select_nearest_propeller_dataset(n_rev_s)
    J = axial_speed_mps / (n_rev_s * D)

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


def aircraft_performance(n_rev_s, axial_speed_mps, rho):
    """Total aircraft propulsive performance from per-rotor performance."""
    per = propeller_performance_per_rotor(n_rev_s, axial_speed_mps, rho)
    total = dict(per)
    total["T_total_N"] = num_rotors * per["T_per_rotor_N"]
    total["P_shaft_total_W"] = num_rotors * per["P_shaft_per_rotor_W"]
    return total


def required_total_thrust(V_vertical, vertical_climb_factor, rho):
    """Steady vertical climb force balance: thrust = weight + optional body drag."""
    weight = MTOM * g
    body_drag = 0.5 * rho * CD_frame * S_ref * V_vertical**2
    return vertical_climb_factor * weight + body_drag


def solve_rotor_speed_for_thrust(V_vertical, vertical_climb_factor, rho, n_low=1.0, n_high=n_max):
    """Bisection solve for rotor speed that gives required steady-climb thrust."""
    target_T = required_total_thrust(V_vertical, vertical_climb_factor, rho)

    T_low = aircraft_performance(n_low, V_vertical, rho)["T_total_N"]
    T_high = aircraft_performance(n_high, V_vertical, rho)["T_total_N"]

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
        T_mid = aircraft_performance(n_mid, V_vertical, rho)["T_total_N"]

        if T_mid < target_T:
            n_low = n_mid
        else:
            n_high = n_mid

    n_solution = 0.5 * (n_low + n_high)
    perf = aircraft_performance(n_solution, V_vertical, rho)
    perf["T_required_total_N"] = target_T
    perf["body_drag_N"] = target_T - MTOM * g
    return perf


# --- 4) Motor, ESC, battery model ---
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
    4015.0: data_4015_rpm,
    5024.0: data_5024_rpm,
    6023.0: data_6023_rpm,
    6024.0: data_6024_rpm,
    7020.0: data_7020_rpm,
    7021.0: data_7021_rpm,
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
def full_powertrain_result(V_vertical, vertical_climb_factor, duty, mission_duration):
    prop = solve_rotor_speed_for_thrust(V_vertical, vertical_climb_factor, rho_hover)
    motor = motor_from_shaft_power(prop["P_shaft_per_rotor_W"], prop["n_rev_s"])
    ess = esc_battery_hydrogen_from_motor(motor, duty, mission_duration)

    result = {}
    result.update(prop)
    result.update(motor)
    result.update(ess)
    result["V_vertical_mps"] = V_vertical
    return result


def print_result(title, result):
    print(f"\n--- {title} ---")
    print(f"Altitude: {hover_altitude:.1f} m")
    print(f"Air density: {rho_hover:.4f} kg/m^3")
    print(f"Vertical speed: {result['V_vertical_mps']:.3f} m/s")
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


hover = full_powertrain_result(V_vertical=0.0, vertical_climb_factor=1.0, 
                               duty=0.45, mission_duration=t_hover)
vertical_climb = full_powertrain_result(V_vertical=V_climb, vertical_climb_factor=2.0, 
                               duty=0.69, mission_duration=t_vertical_climb)

print_result("Hover Performance", hover)
print_result("Vertical Climb Performance", vertical_climb)
