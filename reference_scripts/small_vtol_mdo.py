# 1) --- IMPORTS ---
# Generic
import math
import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import aerosandbox as asb
import aerosandbox.numpy as np

# Physics models
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover_asb, power_cruise_asb, power_climb_asb
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_wing, M_fuselage, M_battery

# GA library (pymoo)
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# 2) --- PARAMETERS ---
t_hover = 120 / 3600       # [h]
n = 1                      # ultimate load factor [-]
l_f = 2.0                  # [m]
w_f = 0.5
d_f = 0.5
num_rotors = 4
num_rotors_hover = 4
num_rotors_cruise = 2
sweep = 0
taper_ratio = 1
t_over_c = 0.12
range_cruise = 5e3 # [m]
CD0 = 0.020
e = 0.85
b_cruise = 1.0
b_climb = 1.0
h_cruise = 80                        # [m]
h_climb = 20                          # [m]
theta_climb = np.radians(10)           # [rad]
h_hover = 20                          # [m]
rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
rho_hover = float(np.squeeze(Atmosphere(h_hover).density))
rho_sl = float(np.squeeze(Atmosphere(0).density))
g = 9.81
eta_elec = 0.9
eta_prop_hover = 0.7
eta_prop_cruise = 0.8
eta_hover = eta_elec * eta_prop_hover
eta_cruise = eta_elec * eta_prop_cruise
k_rotor = 1.5

v_cell = 3.7 # battery cell voltage is 3.7 V
n_cell = 6 # number of cells in series is 6
rho_batt = 180.0 # battery energy density [Wh/kg]
e_usable = 0.80

kv_mot = 1300 # motor velocity constant [RPM/V]
rm_mot = 0.076 # motor internal resistance [Ohms]
rho_motor = 5000 # power density of BLDC motor [W/kg]
io_mot = 0.95 # motor idle current is [A]
P_mot_max = 1059 # motor max power is [W]
v_mot_rated = 22.2 # motor rated voltage (5-6S LiPo) [V]
i_mot_peak = 45.1 # motor peak current [A]

# Mass constants
m_motor = 0.0466 # mass of BLDC motor [kg]
m_payload = 0.250 # mass of payload [kg]
m_flight_controller = 0.073 # mass of cube orange [kg]
m_flight_computer = 0.080 # mass of raspberry pi [kg]
m_avionics = 0.25 # estimated total mass of GPS + remote ID + telemetry module + lidar + receiver + camera + video transmitter [kg]
m_electronics = 0.25 # estimated total mass of ESCs + servos + cables + power modules [kg]


# 3) --- Optimization Parameters ---
opti = asb.Opti()

MTOM = opti.variable(init_guess=10, log_transform=True, lower_bound=1, upper_bound=25)
S = opti.variable(init_guess=1, log_transform=True, lower_bound=0.1, upper_bound=10)
V_cruise = opti.variable(init_guess=10, log_transform=True, lower_bound=1, upper_bound=50)
V_climb = opti.variable(init_guess=10, log_transform=True, lower_bound=1, upper_bound=50)
AR = opti.variable(init_guess=4, log_transform=True, lower_bound=1, upper_bound=10)
R_cruise = opti.variable(init_guess=2*0.0254, log_transform=True, lower_bound=1*0.0254, upper_bound=4*0.0254)
R_hover = opti.variable(init_guess=2*0.0254, log_transform=True, lower_bound=1*0.0254, upper_bound=4*0.0254)


# 4) --- Models ---
# Wing planform processing
b_wing = (S * AR) ** 0.5 # wing span [m]
c = S/b_wing # wing mac [m]
wing_loading = (MTOM * g) / S # [N/m^2]

# Aerodynamic processing
K = 1 / (math.pi * e * AR)
CL_cruise = (2 * b_cruise * MTOM * g) / (rho_cruise * S * V_cruise**2)
CL_climb = (2 * b_climb * MTOM * g) / (rho_climb * S * V_climb**2)
CD_cruise = CD0 + K * CL_cruise**2
CD_climb = CD0 + K * CL_climb**2
CL_over_CD_cruise = CL_cruise / CD_cruise
CL_over_CD_climb = CL_climb / CD_climb

# Power processing
P_hover = 1.15 * power_hover_asb(MTOM=MTOM, rho=rho_hover, R_prop=R_hover, num_rotors=num_rotors_hover, eta_h=eta_hover)
P_cruise = power_cruise_asb(MTOM=MTOM, b_cruise=b_cruise, AR=AR, e=e, CD0=CD0, rho_cruise=rho_cruise, S=S, V_cruise=V_cruise, R_cruise=R_cruise, eta_cruise=eta_cruise, num_rotors_cruise=num_rotors_cruise)
P_climb = power_climb_asb(MTOM=MTOM, b_climb=b_climb, AR=AR, e=e, CD0=CD0, rho_climb=rho_climb, S=S, V_climb=V_climb, theta_climb=theta_climb, R_cruise=R_cruise, eta_cruise=eta_cruise, num_rotors_cruise=num_rotors_cruise)

# Time processing
t_cruise = (range_cruise / V_cruise) / 3600
t_climb = ((h_cruise - h_climb) / (V_climb * math.sin(theta_climb))) / 3600 # hours
t_trip = t_cruise + t_climb + t_hover

# Energy processing
energy_hover = P_hover * t_hover
energy_cruise = P_cruise * t_cruise
energy_climb = P_climb * t_climb
energy_trip = energy_hover + energy_climb + energy_cruise

# Mass processing 
mass_battery = M_battery(E_trip=energy_trip, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
mass_motor = num_rotors * m_motor
mass_wing = M_wing(b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio, t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_payload = m_payload
mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_total = mass_motor + mass_wing + mass_payload + mass_fuselage + mass_battery + m_avionics + m_electronics + m_flight_controller + m_flight_computer


# 5) --- Constraints ---
opti.subject_to([
    MTOM <= 25,
    MTOM == mass_total,
    V_cruise > 5,
    V_climb > 5,
    b_wing <= 2.5,
])


# 6) --- Optimization objective ---
opti.minimize(energy_trip) 


# 7) --- Solve ---
sol = opti.solve(max_iter=300)


# 8) --- Display Results ---
print("--- Optimization Variables ---")
print(f"Optimal V_cruise: {sol.value(V_cruise)} m/s")
print(f"Opimal V_climb: {sol.value(V_climb)} m/s")
print(f"Optimal MTOM: {sol.value(MTOM)} kg")
print(f"Optimal Hover Propeller Radius: {sol.value(R_hover)} m")
print(f"Optimal Cruise Propeller Radius: {sol.value(R_cruise)} m")
print(f"Optimal Wing Area: {sol.value(S)} m^2")
print(" ")
print("--- Wing Planform Properties ---")
print(f"Wingspan: {sol.value(b_wing)} m")
print(f"Wing chord length: {sol.value(c)} m")
print(f"Wing loading: {sol.value(wing_loading)} N/m^2")
print(" ")
print(" --- Aerodynamic Properties ---")
print(f"CL at cruise: {sol.value(CL_cruise)}")
print(f"CL/CD at cruise: {sol.value(CL_over_CD_cruise)}")
print(f"CL at climb: {sol.value(CL_climb)}")
print(f"CL/CD at climb: {sol.value(CL_over_CD_climb)}")
print(" ")
print("--- Thrust and Power Properties ---")
print(f"Total hover power required: {sol.value(P_hover)} W")
print(f"Total cruise power required: {sol.value(P_cruise)} W")
print(f"Total climb power required: {sol.value(P_climb)} W")
print(" ")
print("--- Energy Properties ---")
print(f"Total trip energy required: {sol.value(energy_trip)} Wh")
print(" ")
print("--- Time Properties ---")
print(f"Total cruise time: {sol.value(t_cruise * 60)} mins")
print(f"Total climb time: {sol.value(t_climb * 60)} mins")
print(f"Total trip time: {sol.value(t_trip * 60)} mins")
print(" ")
print("--- Mass Properties ---")
print(f"Total motor mass: {sol.value(mass_motor * 1000)} g")
print(f"Total payload mass: {sol.value(mass_payload * 1000)} g")
print(f"Total wing mass: {sol.value(mass_wing * 1000)} g")
print(f"Total fuselage mass: {sol.value(mass_fuselage * 1000)} g")
print(f"Total battery mass: {sol.value(mass_battery * 1000)} g")
print(f"Maximum takeoff mass: {sol.value(mass_total * 1000)} g")