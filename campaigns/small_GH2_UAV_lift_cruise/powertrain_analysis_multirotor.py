""" 
This script computes the power the energy storage system, ESCs, and motors must produce 
to sustain hover and constant speed vertical climb. 

References:
[1] Improvement of Electric Propulsion System Model for Performance Analysis of Large-Size Multicopter UAVs.
[2] AER1216 Lecture Notes
[3] Data for 20 x 10 propeller gathered from: https://m-selig.ae.illinois.edu/props/volume-4/propDB-volume-4.html

"""


import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import math


# --- 1) Define Component Parameters ---
# Propeller parameters 
D = 29 * 0.0254 # propeller diameter [m]
CTO = 0.1 # static thrust constant at J = 0
CPO = 0.03 # static power constant at J = 0

data_2020_rpm = {
    "J": [
        0.158832, 0.200992, 0.237297, 0.272711,
        0.310842, 0.342752, 0.373119, 0.410827,
        0.449933, 0.483772, 0.527929, 0.559306,
        0.597926, 0.629733, 0.633721, 0.633721
    ],
    "CT": [
        0.088917, 0.083853, 0.079386, 0.071155,
        0.063099, 0.055265, 0.048647, 0.041327,
        0.034095, 0.027493, 0.019174, 0.014299,
        0.006723, -0.000444, -0.000443, -0.000443
    ],
    "CP": [
        0.032303, 0.032564, 0.032891, 0.031759,
        0.030154, 0.027946, 0.025939, 0.023606,
        0.021011, 0.018476, 0.015601, 0.013812,
        0.010693, 0.007450, 0.007542, 0.007542
    ],
    "eta": [
        0.437192, 0.517541, 0.572744, 0.610993,
        0.650448, 0.677798, 0.699761, 0.719238,
        0.730086, 0.719873, 0.648834, 0.579022,
        0.375882, -0.037532, -0.037252, -0.037252
    ]
}

data_2979_rpm = {
    "J": [
        0.124122, 0.146628, 0.176025, 0.199283, 0.223488,
        0.247610, 0.273034, 0.298709, 0.323408, 0.348309,
        0.372103, 0.394564, 0.418023, 0.440591, 0.461942
    ],
    "CT": [
        0.091939, 0.090385, 0.087792, 0.086122, 0.083856,
        0.081130, 0.077595, 0.073883, 0.069559, 0.064618,
        0.059202, 0.053741, 0.048085, 0.043070, 0.037405
    ],
    "CP": [
        0.032246, 0.032511, 0.032725, 0.033094, 0.033381,
        0.033501, 0.033376, 0.033148, 0.032652, 0.031761,
        0.030372, 0.028833, 0.027028, 0.025440, 0.023289
    ],
    "eta": [
        0.353888, 0.407640, 0.472216, 0.518595, 0.561420,
        0.599633, 0.634760, 0.665788, 0.688958, 0.708640,
        0.725311, 0.735432, 0.743685, 0.745914, 0.741940
    ]
}

data_3049_rpm = {
    "J": [
        0.398117, 0.420829, 0.443421, 0.466859,
        0.488826, 0.512623, 0.538637, 0.561638,
        0.587413, 0.610489, 0.634160, 0.634103,
        0.634103, 0.634103, 0.634103, 0.634103,
        0.634103, 0.634103, 0.634103, 0.634103,
        0.634103, 0.634103, 0.634103, 0.634103
    ],
    "CT": [
        0.053316, 0.047333, 0.041681, 0.036106,
        0.031845, 0.026509, 0.021748, 0.017221,
        0.012480, 0.007348, 0.002071, 0.002061,
        0.002061, 0.002061, 0.002061, 0.002061,
        0.002061, 0.002061, 0.002061, 0.002061,
        0.002061, 0.002061, 0.002061, 0.002061
    ],
    "CP": [
        0.028659, 0.026695, 0.024718, 0.022701,
        0.021101, 0.019023, 0.017062, 0.015151,
        0.013169, 0.010830, 0.008328, 0.008311,
        0.008311, 0.008311, 0.008311, 0.008311,
        0.008311, 0.008311, 0.008311, 0.008311,
        0.008311, 0.008311, 0.008311, 0.008311
    ],
    "eta": [
        0.740631, 0.746187, 0.747727, 0.742562,
        0.737697, 0.714341, 0.686579, 0.638396,
        0.556695, 0.414186, 0.157693, 0.157233,
        0.157233, 0.157233, 0.157233, 0.157233,
        0.157233, 0.157233, 0.157233, 0.157233,
        0.157233, 0.157233, 0.157233, 0.157233
    ]
}

data_3958_rpm = {
    "J": [
        0.129821, 0.142446, 0.163081, 0.180469, 0.202024,
        0.217746, 0.238239, 0.254780, 0.272571, 0.290786,
        0.309710, 0.328959, 0.346067
    ],
    "CT": [
        0.095848, 0.095509, 0.093301, 0.091764, 0.089556,
        0.087979, 0.086028, 0.083670, 0.081718, 0.079615,
        0.076979, 0.074180, 0.071474
    ],
    "CP": [
        0.033538, 0.033892, 0.033922, 0.034097, 0.034210,
        0.034271, 0.034418, 0.034277, 0.034313, 0.034312,
        0.034150, 0.033897, 0.033543
    ],
    "eta": [
        0.371005, 0.401419, 0.448544, 0.485688, 0.528859,
        0.558980, 0.595485, 0.621922, 0.649130, 0.674724,
        0.698126, 0.719896, 0.737404
    ]
}

data_4025_rpm = {
    "J": [
        0.291872, 0.310398, 0.328268, 0.347340,
        0.365571, 0.383483, 0.400957, 0.418299,
        0.433539, 0.452953, 0.470074, 0.489575,
        0.505769, 0.526643, 0.544202, 0.563218,
        0.584476, 0.602380, 0.619334, 0.639872,
        0.655653, 0.655462, 0.655462, 0.655462
    ],
    "CT": [
        0.079187, 0.076958, 0.074193, 0.071416,
        0.068536, 0.065419, 0.062011, 0.058348,
        0.054693, 0.050533, 0.046168, 0.041728,
        0.036660, 0.031702, 0.027194, 0.022108,
        0.017646, 0.013273, 0.009459, 0.005299,
        0.001464, 0.001451, 0.001451, 0.001451
    ],
    "CP": [
        0.034141, 0.034074, 0.033747, 0.033431,
        0.033020, 0.032452, 0.031704, 0.030703,
        0.029614, 0.028331, 0.026929, 0.025543,
        0.023634, 0.021742, 0.019894, 0.017732,
        0.015682, 0.013586, 0.011790, 0.009816,
        0.007838, 0.007832, 0.007832, 0.007832
    ],
    "eta": [
        0.676960, 0.701057, 0.721711, 0.741986,
        0.758767, 0.773040, 0.784254, 0.794944,
        0.800703, 0.807900, 0.805922, 0.799787,
        0.784525, 0.767903, 0.743920, 0.702187,
        0.657660, 0.588539, 0.496890, 0.345442,
        0.122501, 0.121457, 0.121457, 0.121457
    ]
}

data_4506_rpm = {
    "J": [
        0.257544, 0.271600, 0.288206, 0.306338,
        0.321786, 0.338747, 0.356564, 0.372573,
        0.388613, 0.404398, 0.418644, 0.433247,
        0.447601, 0.463198, 0.479032, 0.496369,
        0.513081, 0.529811, 0.545398, 0.569711,
        0.582643, 0.598384, 0.613118, 0.636422
    ],
    "CT": [
        0.085100, 0.083269, 0.081441, 0.079450,
        0.076397, 0.074090, 0.071641, 0.068948,
        0.066418, 0.063626, 0.060887, 0.058136,
        0.054974, 0.051048, 0.047177, 0.043292,
        0.038986, 0.034816, 0.030820, 0.024491,
        0.021307, 0.016585, 0.013616, 0.008203
    ],
    "CP": [
        0.034759, 0.034695, 0.034734, 0.034778,
        0.034271, 0.034100, 0.033853, 0.033407,
        0.032975, 0.032411, 0.031763, 0.031033,
        0.030115, 0.028885, 0.027685, 0.026436,
        0.024876, 0.023232, 0.021618, 0.018985,
        0.017589, 0.015377, 0.013937, 0.011308
    ],
    "eta": [
        0.630527, 0.651855, 0.675765, 0.699829,
        0.717327, 0.736016, 0.754563, 0.768958,
        0.782735, 0.793881, 0.802485, 0.811624,
        0.817083, 0.818591, 0.816289, 0.812833,
        0.804106, 0.793996, 0.777552, 0.734935,
        0.705799, 0.645386, 0.599005, 0.461634
    ]
}
n_max = 100

# ESC parameters
r_esc_on = 2.2e-3# esc on resistance [Ohm]
rise_time = 5.7e-9 # rise time [sec]
fall_time = 11.1e-9 # fall time [sec]
f_sw = 16e3 # switching frequncy in Hz
P_ic = 0.5 # power consumption of the ESC IC [W]
num_switches_effective = 6.0

# BLDC Parameters
Kv = 100 * 2 * math.pi/60 # motor velocity constant [rad/s/V]
Kt = 1/Kv # motor torque constant
i0 = 1.4 # motor no load current [A]
rm = 0.065 # motor internal resistance [Ohms]

# LiPo Parameters
num_cells = 12
cell_voltage = 3.8 # nominal cell voltage of a LiPo

# Hydrogen System Parameters
eta_fc = 0.5 # fuel cell efficiency 
LHV_H2 = 33333 # [Wh/kg]

# UAV parameters 
MTOM = 21 # total mass [kg]
g = 9.81 # acceleration due to gravity [m*s^-2]
hover_altitude = 10 # [m]
rho_hover = float(np.squeeze(Atmosphere(hover_altitude).density)) # air density at hover altitude [kg/m^3]
num_rotors = 4 # number of rotors for hover [-]
V_climb = 1.5 # maximum climb speed [m/s]
CD_frame = 0.5 # body drag coefficient [-]
S_ref = 0.9 # reference area for drag computation [m^2]

# Mission Parameters
t_hover = 10 * (1/60) # hours 
t_vertical_climb = 1 * (1/60) # hours
n_hybrid_factor_fc_hover = 0.4
n_hybrid_factor_fc_vertical_climb = 0.2


# --- 2) Helper Functions
def get_coeff_at_J_interp(propeller_data, target_J, coeff_name):

    """ Return propeller coefficient from linear interpolation at target advance ratio."""
    
    propeller_data["J"].insert(0, 0.0)
    propeller_data["CT"].insert(0, CTO)
    propeller_data["CP"].insert(0, CPO)
    propeller_data["eta"].insert(0, 0.0)

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


def esc_battery_hydrogen_from_motor(motor, duty, eta_fc, hybrid_factor, mission_duration):

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

    P_fc_required = hybrid_factor * P_bus_total_W
    P_batt_required = (1-hybrid_factor) * P_bus_total_W

    rH2 = P_fc_required / (eta_fc * LHV_H2)
    mass_H2 = rH2 * mission_duration

    C_batt = (P_batt_required * mission_duration) / (V_bus_ideal)

    return {
        "V_bus_V": V_bus_ideal,
        "P_esc_loss_per_rotor_W": P_esc_loss,
        "P_bus_per_rotor_W": P_bus_per_rotor,
        "I_bus_per_rotor_A": I_bus_per_rotor,
        "P_bus_total_W": num_rotors * P_bus_per_rotor,
        "I_bus_total_A": num_rotors * I_bus_per_rotor,
        "P_fc_required_W": P_fc_required,
        "P_batt_required_W": P_batt_required,
        "mass_H2_g": mass_H2 * 1000,
        "C_batt_required_Ah": C_batt
    }


# --- 3) Data Visualization ---
propeller_datasets = {
    "2020 RPM": data_2020_rpm,
    "2979 RPM": data_2979_rpm,
    "3049 RPM": data_3049_rpm,
    "3958 RPM": data_3958_rpm,
    "4025 RPM": data_4025_rpm,
    "4506 RPM": data_4506_rpm,
}

propeller_datasets = {
    2020.0: data_2020_rpm,
    2979.0: data_2979_rpm,
    3049.0: data_3049_rpm,
    3958.0: data_3958_rpm,
    4025.0: data_4025_rpm,
    4506.0: data_4506_rpm,
}


fit_variables = {
    "CT": "Thrust coefficient, $C_T$",
    "CP": "Power coefficient, $C_P$",
    "eta": "Efficiency, $\\eta$",
}

for propeller_data in propeller_datasets.values():
    if propeller_data["J"][0] != 0.0:
        propeller_data["J"].insert(0, 0.0)
        propeller_data["CT"].insert(0, CTO)
        propeller_data["CP"].insert(0, CPO)
        propeller_data["eta"].insert(0, 0.0)

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
def full_powertrain_result(V_vertical, vertical_climb_factor, duty, hybrid_factor, mission_duration, eta_fc):
    prop = solve_rotor_speed_for_thrust(V_vertical, vertical_climb_factor, rho_hover)
    motor = motor_from_shaft_power(prop["P_shaft_per_rotor_W"], prop["n_rev_s"])
    ess = esc_battery_hydrogen_from_motor(motor, duty, eta_fc, hybrid_factor, mission_duration)

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
    print(f"Fuel cell power required: {result['P_fc_required_W']:.1f} W")
    print(f"Battery power required: {result['P_batt_required_W']:.1f} W"),
    print(f"Mass of hydrogen required: {result['mass_H2_g']:.1f} g")
    print(f"Battery capacity required: {result['C_batt_required_Ah']:.1f} Ah")


hover = full_powertrain_result(V_vertical=0.0, vertical_climb_factor=1.0, 
                               duty=0.50, hybrid_factor=n_hybrid_factor_fc_hover, 
                               mission_duration=t_hover, eta_fc=eta_fc)
vertical_climb = full_powertrain_result(V_vertical=V_climb, vertical_climb_factor=2.0, 
                               duty=0.76, hybrid_factor=n_hybrid_factor_fc_vertical_climb,
                               mission_duration=t_vertical_climb, eta_fc=eta_fc)

print_result("Hover Performance", hover)
print_result("Vertical Climb Performance", vertical_climb)