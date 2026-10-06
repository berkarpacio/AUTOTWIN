import math
from tests.test_flow_conditions import flow_conditions
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np

# 7 x 6 Hover Propeller Data, available at: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html 
CT0 = 0.09
CP0 = 0.07
D_prop = 7 * 0.0254 # propeller diameter [m]
eta_prop_hover = 0.7
N_rotors_hover = 4

J = [0.584, 0.620, 0.656, 0.688, 0.721, 0.752, 0.796,
    0.822, 0.849, 0.891, 0.930, 0.952, 0.979, 1.040]

CT = [0.0579, 0.0531, 0.0472, 0.0425, 0.0378, 0.0327, 0.0252,
    0.0209, 0.0170, 0.0112, 0.0049, 0.0016, -0.0036, -0.0139]

CP = [0.0591, 0.0568, 0.0533, 0.0505, 0.0474, 0.0444, 0.0400,
    0.0369, 0.0338, 0.0293, 0.0249, 0.0216, 0.0186, 0.0109]

eta = [0.572, 0.580, 0.581, 0.578, 0.575, 0.553, 0.503,
    0.465, 0.427, 0.341, 0.184, 0.070, -0.192, -1.335]

# 7 x 6 Cruise Propeller Data, available at: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html 
CT0 = 0.09
CP0 = 0.07
D_prop = 7 * 0.0254 # propeller diameter [m]
N_rotors_cruise = 1

J = [0.584, 0.620, 0.656, 0.688, 0.721, 0.752, 0.796,
    0.822, 0.849, 0.891, 0.930, 0.952, 0.979, 1.040]

CT = [0.0579, 0.0531, 0.0472, 0.0425, 0.0378, 0.0327, 0.0252,
    0.0209, 0.0170, 0.0112, 0.0049, 0.0016, -0.0036, -0.0139]

CP = [0.0591, 0.0568, 0.0533, 0.0505, 0.0474, 0.0444, 0.0400,
    0.0369, 0.0338, 0.0293, 0.0249, 0.0216, 0.0186, 0.0109]

eta = [0.572, 0.580, 0.581, 0.578, 0.575, 0.553, 0.503,
    0.465, 0.427, 0.341, 0.184, 0.070, -0.192, -1.335]


# BLDC Data
Kv = 1300 * 2 * math.pi/60 # motor velocity constant [rad/s/V]
Kt = 1/Kv # motor torque constant
i0 = 0.95 # motor no load current [A]
rm = 0.076 # motor internal resistance [Ohms]

# ESC Data
r_esc_on = 2.2e-3# esc on resistance [Ohm]
rise_time = 5.7e-9 # rise time [sec]
fall_time = 11.1e-9 # fall time [sec]
f_sw = 16e3 # switching frequncy in Hz
P_ic = 0.5 # power consumption of the ESC IC [W]
num_switches_effective = 6.0

# Battery Data
N_cells = 6
v_cell = 3.8
V_bus = N_cells * v_cell
C_batt_max = 8

# UAV data 
MTOM = 3
c_ref = 0.190
CL_cruise = 0.6
CD_cruise = 0.030
S = 0.303

# Mission data 
h_hover = 10
h_cruise = 90
g = 9.81
t_hover = 5 # minutes
t_cruise = 40 # minutes
V_cruise = 16 # m/s

# Utility function for solving for n when J  is non-zero
def determine_n(CT: list, J: list, V: float, T: float, h:float) -> float:
    """
    Determines propeller rotational speed n [rev/s] required
    to generate thrust T at forward/axial velocity V.

    Inputs:
        CT -> thrust coefficient data [-]
        CP -> power coefficient data [-] (not used for solving n)
        J  -> advance ratio data [-]
        V  -> air velocity through propeller [m/s]
        T  -> required thrust per propeller [N]

    Global variables required:
        rho    -> air density [kg/m^3]
        D_prop -> propeller diameter [m]

    Returns:
        n -> rotational speed [rev/s]
    """

    # Get flow conditions
    _, _, rho = flow_conditions(h=h, c_ref=c_ref)

    J = np.asarray(J)
    CT = np.asarray(CT)

    def thrust_residual(n):

        J_op = V / (n * D_prop)

        # Prevent extrapolation outside available J data
        if J_op < J[0] or J_op > J[-1]:
            raise ValueError(
                f"Operating J = {J_op:.3f} is outside "
                f"the available range [{J[0]:.3f}, {J[-1]:.3f}]"
            )

        CT_op = np.interp(J_op, J, CT)

        T_calc = CT_op * rho * n**2 * D_prop**4

        return T_calc - T

    # n corresponding to maximum and minimum J in dataset
    n_min = V / (J[-1] * D_prop)
    n_max = V / (J[0] * D_prop)

    n = brentq(
        thrust_residual,
        n_min,
        n_max
    )

    return n

# Utility function for solving for shaft power when J is non-zero
def shaft_power_from_n(CP: list, J: list, V: float, n: float, h: float) -> float:

    """
    Determines shaft power of one propeller using the solved
    rotational speed n.

    Inputs:
        CP -> power coefficient data [-]
        J  -> advance ratio data [-]
        V  -> axial / flight velocity [m/s]
        n  -> propeller rotational speed [rev/s]

    Global variables required:
        rho    -> air density [kg/m^3]
        D_prop -> propeller diameter [m]

    Returns:
        P_shaft -> shaft power per propeller [W]
    """

    # Get flow conditions
    _, _, rho = flow_conditions(h=h, c_ref=c_ref)

    J_data = np.asarray(J)
    CP_data = np.asarray(CP)

    # Determine operating advance ratio
    J_op = V / (n * D_prop)

    # Check whether operating point is within data range
    if J_op < J_data[0] or J_op > J_data[-1]:
        raise ValueError(
            f"Operating J = {J_op:.3f} is outside "
            f"available data range "
            f"[{J_data[0]:.3f}, {J_data[-1]:.3f}]"
        )

    # Interpolate power coefficient
    CP_op = np.interp(J_op, J_data, CP_data)

    # Shaft power
    P_shaft = (CP_op * rho * n**3 * D_prop**5)

    return P_shaft


def hover_power_required_empirical(CT0, CP0, D_prop, h_hover, N_rotors_hover, MTOM, g):

    # Get flow conditions
    _, _, rho_hover = flow_conditions(h=h_hover, c_ref=c_ref)

    # Compute power
    n_hover = math.sqrt((MTOM * g / N_rotors_hover) * 1/(CT0 * rho_hover * D_prop**4))
    CQ0 = CP0 / (2 * math.pi)
    Q_rotor_shaft = CQ0 * rho_hover * n_hover**2 * D_prop**5
    P_rotor_shaft = Q_rotor_shaft * 2 * math.pi * n_hover

    return P_rotor_shaft, Q_rotor_shaft


def hover_power_required_momentum_theory(h_hover, D_prop, eta_prop_hover, N_rotors_hover, MTOM, g) -> float:

    # Get flow conditions
    _, _, rho_hover = flow_conditions(h=h_hover, c_ref=c_ref)

    # Compute power
    T_rotor = (MTOM * g) / N_rotors_hover
    A_rotor = math.pi * (D_prop/2)**2
    disk_loading = T_rotor / A_rotor
    power_rotor = T_rotor * math.sqrt(disk_loading/(2 * rho_hover))
    P_rotor_shaft  = power_rotor / eta_prop_hover

    return P_rotor_shaft


def cruise_power_required_empirical(CD_cruise, h_cruise, S, V_cruise, CP, J):

    _, _, rho = flow_conditions(h=h_cruise, c_ref=c_ref)
    T_cruise = 0.5 * CD_cruise *  rho * S * V_cruise**2
    n_cruise = determine_n(CT, J, V_cruise, T_cruise, h_cruise)

    P_rotor_shaft = shaft_power_from_n(CP, J, V_cruise, n_cruise, h_cruise)
    Q_rotor_shaft = P_rotor_shaft / (2 * math.pi * n_cruise)

    return P_rotor_shaft, Q_rotor_shaft


# For each mission segment, computes battery capacity required
def capacity_required(P_rotor_shaft, Q_rotor_shaft, Kv, i0, rm, V_bus, 
                      P_ic, rise_time, fall_time, r_esc_on, f_sw, 
                      num_switches_effective, N_rotors, t_mission):

    # BLDC model
    I_motor = Q_rotor_shaft * Kv + i0
    omega = P_rotor_shaft / Q_rotor_shaft
    V_motor = omega / Kv + I_motor * rm
    P_motor_elec = V_motor * I_motor
    eta_motor = P_rotor_shaft / P_motor_elec

    # ESC model
    duty = V_motor/V_bus
    P_cond = duty * I_motor**2 * r_esc_on
    P_sw = 0.5 * V_bus * I_motor * (rise_time + fall_time) * f_sw * num_switches_effective
    P_esc_loss = P_cond + P_sw + P_ic
    P_bus_rotor = P_motor_elec + P_esc_loss

    # Battery mode
    I_bus_rotor = P_bus_rotor / V_bus
    I_batt_total = I_bus_rotor * N_rotors
    C_batt_mission = (t_mission/60) * I_batt_total

    return C_batt_mission, duty


if __name__ == "__main__":

    # 1) Hover Mission Segment
    P_rotor_shaft_hover, Q_rotor_shaft_hover = hover_power_required_empirical(CT0, CP0, D_prop, h_hover, N_rotors_hover, MTOM, g)
    P_rotor_shaft_cruise, Q_rotor_shaft_cruise = cruise_power_required_empirical(CD_cruise, h_cruise, S, V_cruise, CP, J)

    # 2) Cruise Mission Segment
    C_battery_hover, duty_hover = capacity_required(P_rotor_shaft_hover, Q_rotor_shaft_hover, Kv, i0, rm, V_bus, P_ic, rise_time, fall_time, r_esc_on, f_sw, num_switches_effective, N_rotors_hover, t_hover)
    C_battery_cruise, duty_cruise = capacity_required(P_rotor_shaft_cruise, Q_rotor_shaft_cruise, Kv, i0, rm, V_bus, P_ic, rise_time, fall_time, r_esc_on, f_sw, num_switches_effective, N_rotors_cruise, t_cruise)

    

    print(P_rotor_shaft_hover)
    print(P_rotor_shaft_cruise)
    print(C_battery_hover)
    print(C_battery_cruise)
    print(duty_cruise)
    print(duty_hover)

    # Aircraft design pass/fail criteria 
    if C_battery_cruise + C_battery_hover <= C_batt_max and duty_cruise < 1 and duty_hover < 1:
        print("Design PASS")
    else: 
        print("Design FAIL")
