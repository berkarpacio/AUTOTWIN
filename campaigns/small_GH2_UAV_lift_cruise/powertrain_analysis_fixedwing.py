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
D = 18 * 0.0254 # propeller diameter [m]
CTO = 0.11 # static thrust constant at J = 0
CPO = 0.04 # static power constant at J = 0

data_2012_rpm = prop_data = {
    "J": [
        0.186616, 0.230077, 0.271963, 0.313014,
        0.353661, 0.391001, 0.426045, 0.461453,
        0.498383, 0.541334, 0.578461, 0.625462,
        0.663311, 0.701471, 0.736429, 0.736580
    ],
    "CT": [
        0.100195, 0.096009, 0.092191, 0.087521,
        0.081109, 0.072582, 0.064751, 0.056584,
        0.049001, 0.041131, 0.034420, 0.025026,
        0.017930, 0.010280, 0.003520, 0.003470
    ],
    "CP": [
        0.041009, 0.041483, 0.042064, 0.042249,
        0.041742, 0.039878, 0.037861, 0.035050,
        0.032170, 0.028899, 0.025929, 0.021332,
        0.017826, 0.013802, 0.010125, 0.010095
    ],
    "eta": [
        0.455947, 0.532496, 0.596058, 0.648423,
        0.687208, 0.711673, 0.728633, 0.744955,
        0.759141, 0.770461, 0.767862, 0.733777,
        0.667160, 0.522460, 0.255964, 0.253160
    ]
}

data_2973_rpm = prop_data = {
    "J": [
        0.143220, 0.171715, 0.200086, 0.225825, 0.253417,
        0.280552, 0.309164, 0.335974, 0.362801, 0.391775,
        0.418847, 0.446243, 0.470696, 0.496097, 0.521775
    ],
    "CT": [
        0.104363, 0.101637, 0.100140, 0.098085, 0.095407,
        0.092262, 0.089082, 0.086247, 0.082271, 0.078970,
        0.074760, 0.070072, 0.064444, 0.058588, 0.053414
    ],
    "CP": [
        0.040743, 0.041129, 0.041844, 0.042324, 0.042541,
        0.042588, 0.042630, 0.042788, 0.042391, 0.042439,
        0.041895, 0.040993, 0.039378, 0.037385, 0.035378
    ],
    "eta": [
        0.366850, 0.424313, 0.478834, 0.523341, 0.568341,
        0.607779, 0.646043, 0.677220, 0.704115, 0.729006,
        0.747417, 0.762775, 0.770312, 0.777465, 0.787780
    ]
}

data_3032_rpm = prop_data = {
    "J": [
        0.449285, 0.474760, 0.500848, 0.525297,
        0.548227, 0.571756, 0.597904, 0.627521,
        0.648636, 0.678472, 0.702087, 0.735282,
        0.757728, 0.787588, 0.789417, 0.789417,
        0.789417, 0.789417, 0.789417, 0.789417,
        0.789417, 0.789417, 0.789417, 0.789417
    ],
    "CT": [
        0.069454, 0.063907, 0.058129, 0.053150,
        0.048059, 0.042240, 0.037181, 0.031779,
        0.026352, 0.020834, 0.015334, 0.010082,
        0.004353, -0.001940, -0.001966, -0.001966,
        -0.001966, -0.001966, -0.001966, -0.001966,
        -0.001966, -0.001966, -0.001966, -0.001966
    ],
    "CP": [
        0.040689, 0.039112, 0.037135, 0.035023,
        0.032962, 0.030325, 0.027980, 0.025500,
        0.022734, 0.019987, 0.017061, 0.014291,
        0.010789, 0.006870, 0.006887, 0.006887,
        0.006887, 0.006887, 0.006887, 0.006887,
        0.006887, 0.006887, 0.006887, 0.006887
    ],
    "eta": [
        0.766903, 0.775738, 0.783992, 0.797175,
        0.799333, 0.796396, 0.794513, 0.782038,
        0.751878, 0.707241, 0.631002, 0.518697,
        0.305706, -0.222466, -0.225314, -0.225314,
        -0.225314, -0.225314, -0.225314, -0.225314,
        -0.225314, -0.225314, -0.225314, -0.225314
    ]
}

data_3958_rpm = prop_data = {
    "J": [
        0.124099, 0.142625, 0.164943, 0.184729,
        0.207582, 0.219941, 0.244969, 0.265818,
        0.286920, 0.308614, 0.329130, 0.349052,
        0.369494, 0.389249
    ],
    "CT": [
        0.109116, 0.106630, 0.106010, 0.104646,
        0.103368, 0.101421, 0.099704, 0.097415,
        0.095227, 0.093129, 0.090847, 0.087908,
        0.085723, 0.082679
    ],
    "CP": [
        0.040992, 0.041076, 0.041867, 0.042299,
        0.042854, 0.042725, 0.043228, 0.043330,
        0.043433, 0.043608, 0.043627, 0.043338,
        0.043360, 0.043010
    ],
    "eta": [
        0.330339, 0.370235, 0.417644, 0.457006,
        0.500708, 0.522104, 0.565015, 0.597615,
        0.629075, 0.659078, 0.685360, 0.708018,
        0.730500, 0.748256
    ]
}

data_4017_rpm = prop_data = {
    "J": [
        0.328521, 0.349689, 0.369763, 0.390427,
        0.409843, 0.429795, 0.450529, 0.469347,
        0.489926, 0.510384, 0.526457, 0.544424,
        0.566126, 0.587573, 0.604277, 0.626523,
        0.646225, 0.669781, 0.687391, 0.706996,
        0.726860, 0.750665, 0.767535, 0.790095
    ],
    "CT": [
        0.090627, 0.088394, 0.085349, 0.083312,
        0.080764, 0.077374, 0.074323, 0.070681,
        0.067513, 0.064265, 0.060287, 0.056448,
        0.052616, 0.048121, 0.044291, 0.039101,
        0.035140, 0.030362, 0.026294, 0.022394,
        0.017686, 0.013028, 0.008907, 0.004127
    ],
    "CP": [
        0.043455, 0.043495, 0.043190, 0.043221,
        0.043002, 0.042424, 0.041954, 0.041114,
        0.040366, 0.039524, 0.038132, 0.036765,
        0.035405, 0.033654, 0.032024, 0.029734,
        0.027917, 0.025444, 0.023289, 0.021254,
        0.018642, 0.016012, 0.013471, 0.010469
    ],
    "eta": [
        0.685140, 0.710670, 0.730706, 0.752574,
        0.769748, 0.783878, 0.798136, 0.806887,
        0.819413, 0.829876, 0.832326, 0.835908,
        0.841327, 0.840154, 0.835737, 0.823870,
        0.813414, 0.799255, 0.776091, 0.744924,
        0.689563, 0.610743, 0.507480, 0.311439
    ]
}

data_4959_rpm = prop_data = {
    "J": [
        0.141950, 0.156221, 0.171562, 0.194248,
        0.208600, 0.223227, 0.237627, 0.254927,
        0.274102, 0.289981, 0.306782
    ],
    "CT": [
        0.110421, 0.109363, 0.108315, 0.106711,
        0.105885, 0.104373, 0.102908, 0.101654,
        0.099532, 0.097717, 0.096251
    ],
    "CP": [
        0.042164, 0.042494, 0.042875, 0.043411,
        0.043804, 0.043918, 0.044073, 0.044439,
        0.044500, 0.044537, 0.044723
    ],
    "eta": [
        0.371730, 0.402043, 0.433417, 0.477491,
        0.504227, 0.530506, 0.554850, 0.583139,
        0.613068, 0.636243, 0.660246
    ]
}

data_4991_rpm = prop_data = {
    "J": [
        0.258570, 0.274564, 0.288863, 0.306373,
        0.323175, 0.339508, 0.356592, 0.373122,
        0.389268, 0.405373, 0.421111, 0.437737,
        0.453480, 0.468809, 0.485439, 0.502105,
        0.515902, 0.529227, 0.544311, 0.562855,
        0.578401, 0.592928, 0.611812, 0.629298
    ],
    "CT": [
        0.100976, 0.099668, 0.097903, 0.096272,
        0.094137, 0.092704, 0.090980, 0.088889,
        0.086598, 0.084474, 0.081771, 0.079633,
        0.077204, 0.074295, 0.072592, 0.069998,
        0.067462, 0.065077, 0.062344, 0.058802,
        0.056012, 0.052847, 0.049289, 0.045475
    ],
    "CP": [
        0.044124, 0.044384, 0.044375, 0.044533,
        0.044479, 0.044660, 0.044728, 0.044655,
        0.044392, 0.044172, 0.043715, 0.043490,
        0.043061, 0.042353, 0.042182, 0.041593,
        0.040827, 0.040091, 0.039229, 0.038015,
        0.037050, 0.035814, 0.034450, 0.032864
    ],
    "eta": [
        0.591725, 0.616557, 0.637313, 0.662321,
        0.683993, 0.704745, 0.725323, 0.742729,
        0.759373, 0.775215, 0.787705, 0.801525,
        0.813046, 0.822371, 0.835414, 0.844999,
        0.852451, 0.859050, 0.865043, 0.870612,
        0.874419, 0.874922, 0.875358, 0.870790
    ]
}
n_max = 8000/60


# ESC parameters
r_esc_on = 2.2e-3# esc on resistance [Ohm]
rise_time = 5.7e-9 # rise time [sec]
fall_time = 11.1e-9 # fall time [sec]
f_sw = 16e3 # switching frequncy in Hz
P_ic = 0.5 # power consumption of the ESC IC [W]
num_switches_effective = 6.0

# BLDC Parameters (https://store.tmotor.com/product/ax525-a-kv250-fixed-wing-motor.html)
Kv = 220 * 2 * math.pi/60 # motor velocity constant [rad/s/V]
Kt = 1/Kv # motor torque constant
i0 = 1.8 # motor no load current [A]
rm = 0.011 # motor internal resistance [Ohms]

# LiPo Parameters
num_cells = 12
cell_voltage = 3.8 # nominal cell voltage of a LiPo

# Hydrogen System Parameters
eta_fc = 0.5 # fuel cell efficiency 
LHV_H2 = 33333 # [Wh/kg]

# UAV parameters 
MTOM = 21 # total mass [kg]
g = 9.81 # acceleration due to gravity [m*s^-2]
altitude = 90 # [m]
hover_altitude = 10 # [m]
rho = float(np.squeeze(Atmosphere(altitude).density)) # air density at hover altitude [kg/m^3]
num_rotors = 1 # number of rotors for hover [-]
V_climb = 13 # maximum climb speed [m/s]
V_cruise = 16 # maximum cruise speed [m/s]
V_stall = 10 # stall speed [m/s]
CD0 = 0.03 # parasitic drag coefficient
AR = 4.0 # wing aspect ratio
CL_cruise = 0.6 # lift coefficient at cruise of 3D wing
CL_max = 1.8 # maximum lift coefficient of 3D wing
S_cruise = (MTOM*g)/(0.5*rho*CL_cruise*V_cruise**2)
S_stall = (MTOM*g)/(0.5*rho*CL_max*V_stall**2)
# S_ref = max(S_cruise, S_stall)
S_ref = S_cruise
b_ref = math.sqrt(S_ref*AR)
e = 0.85 # 3D wing Oswald's efficiency 
theta_climb = 10 # climb angle in degreees

# Mission Parameters
t_cruise = 7 # hours 
t_climb = ((altitude - hover_altitude) / ((math.sin(math.radians(theta_climb))) * V_climb)) / 3600 # hours
print(t_climb)
n_hybrid_factor_fc_cruise = 1.0
n_hybrid_factor_fc_climb = 0.5


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
    "2012 RPM": data_2012_rpm,
    "2973 RPM": data_2973_rpm,
    "3032 RPM": data_3032_rpm,
    "3958 RPM": data_3958_rpm,
    "4017 RPM": data_4017_rpm,
    "4959 RPM": data_4959_rpm,
    "4991 RPM": data_4991_rpm,
}

propeller_datasets = {
    2012.0: data_2012_rpm,
    2973.0: data_2973_rpm,
    3032.0: data_3032_rpm,
    3958.0: data_3958_rpm,
    4017.0: data_4017_rpm,
    4959.0: data_4959_rpm,
    4991.0: data_4991_rpm,
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
def full_powertrain_result(V, theta_climb, duty, hybrid_factor, mission_duration, eta_fc):
    prop = solve_rotor_speed_for_thrust(V, rho, theta_climb)
    motor = motor_from_shaft_power(prop["P_shaft_per_rotor_W"], prop["n_rev_s"])
    battery = esc_battery_hydrogen_from_motor(motor, duty, eta_fc, hybrid_factor, mission_duration)

    result = {}
    result.update(prop)
    result.update(motor)
    result.update(battery)
    result["V_forward_mps"] = V
    return result


def print_result(title, result):
    print(f"\n--- {title} ---")
    print(f"Wing reference area: {S_ref} m^2")
    print(f"Wing span: {b_ref} m")
    print(f"Altitude: {altitude:.1f} m")
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
    print(f"Fuel cell power required: {result['P_fc_required_W']:.1f} W")
    print(f"Battery power required: {result['P_batt_required_W']:.1f} W"),
    print(f"Mass of hydrogen required: {result['mass_H2_g']:.1f} g")
    print(f"Battery capacity required: {result['C_batt_required_Ah']:.1f} Ah")



def run_case(title, V, theta_climb, duty, hybrid_factor, mission_duration, eta_fc):
    try:
        result = full_powertrain_result(V=V, theta_climb=theta_climb, duty=duty, 
        hybrid_factor=hybrid_factor, mission_duration=mission_duration, eta_fc=eta_fc)
    except RuntimeError as exc:
        print(f"\n--- {title} ---")
        print(f"No feasible rotor speed found: {exc}")
        print("Increase n_max, increase prop diameter/count, reduce drag/weight, or reduce the required speed/climb angle.")
        return None

    print_result(title, result)
    return result


cruise = run_case("Cruise Performance", V=V_cruise, theta_climb=0, duty=0.52, 
                  hybrid_factor=n_hybrid_factor_fc_cruise, mission_duration=t_cruise, eta_fc=eta_fc)
climb = run_case("Climb Performance", V=V_climb, theta_climb=theta_climb, duty=0.65, 
                 hybrid_factor=n_hybrid_factor_fc_climb, mission_duration=t_climb, eta_fc=eta_fc)
