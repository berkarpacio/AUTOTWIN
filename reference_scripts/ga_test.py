"""
Multi-objective (energy_trip, time_trip) genetic algorithm optimization (NSGA-II) using pymoo
for a Fan-in-Wing VTOL with a pusher turboprop engine powered by a fuel cell and an electric motor

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ambiance import Atmosphere

# ---- Model imports ----
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover, power_cruise_fuel, power_climb_fuel
from DesignOS.EVTOLMDO.Model_functions.Cost_model import economic_analysis
from DesignOS.EVTOLMDO.Model_functions.Weight_model import (
    M_rotor, M_motor, M_systems, M_payload, M_wing, M_fuselage, M_furnish, M_battery, 
    M_hydrogen_turboprop_cruise, M_fuel_cell, V_hydrogen
)

# ---- GA library (pymoo) ----
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# =========================
# 1) PARAMETERS
# =========================
# Aircraft 
range_cruise = 100e3       # [m]
CD0 = 0.020
t_hover = 120 / 3600       # [h]
# Wing
n = 1                      # ultimate load factor [-]
sweep = 0
taper_ratio = 1
t_over_c = 0.12
e = 0.85
# Fuselage
l_f = 6.0                  # [m]
w_f = 1.0
d_f = 3.0
# Battery
rho_batt = 180.0           # [Wh/kg]
c_rate = 10
e_usable = 0.80
# Electric motor
eta_elec = 0.9
rho_motor = 4225.806       # [W/kg]
# Lift fans
num_rotors = 4
k_rotor = 1.5
eta_prop_hover = 0.7
eta_hover = eta_elec * eta_prop_hover
# Cruise propeller
eta_prop_cruise = 0.85
# Fuel System
rho_fuel_sys = 0.06
rho_fuel = 33000           # [Wh/kg]
LHV = 120e6 # [J/kg]
# Fuel Cell Module
rho_fuel_cell = 860 # W/kg
eta_fc = 0.64
# Payload
m_pax = 82
n_seats = 4
m_lugg = 16
n_crew = 0
# Mission
b_cruise = 0.95
b_climb = 1.0
h_cruise = 1000            # [m]
h_climb = 100              # [m]
theta_climb = 10           # [deg]
h_hover = 100              # [m]
rho_sl = float(np.squeeze(Atmosphere(0).density))
g = 9.81
# Economic
n_wd = 260 # Number of operating days per year of one eVTOL [days]
T_D = 8 # Daily operation window of one eVTOL [-]
C_rate_charge = 2 
P_energy = 0.11 # Electricity price [$/kWh]
j = 0.0796 # Annuity factor [-]
P_empty = 1664.1 # Price of eVTOL per empty weight [$/kg]
x_ins = 0.06 # Insurance factor
x_irs = 0.14 # Reservation and sales factor
x_iap = 0.02 # Advertising and publicity factor
x_iga = 0.06 # General and administration factor
MF = 0.6 # Maintenance man-hours per flight hour ratio
MWR = 63.71 # Maintenance wrap-rate [$/hr]
P_batt = 150.6 # Energy specific battery acquisition cost [$/kWh]
N_cycles = 420 # Number of life cycles available per battery
n_ac = 1 # number of aircraft controlled by one pilot
P_pilot = 45.2 # hourly salary of a pilot [$/hr]
fare = 3.0 # fare per km [$/km]
# Equality constraint tolerance (MTOM == mass_total)
tol_mass_kg = 5.0


MTOM = 4000
S = 55
A_disk = 26
V_cruise = 80
V_climb = 80
AR = 5

# Atmosphere
rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
rho_hover = float(np.squeeze(Atmosphere(h_hover).density))

# Geometry
R_rotor = math.sqrt((A_disk / num_rotors) / math.pi)
b_wing = math.sqrt(S * AR)
c = S / b_wing

# Aerodynamics (same as M_fuel_jet_cruise)
CL_cruise = (2 * b_cruise * MTOM * g) / (rho_cruise * S * V_cruise**2)
K = 1.0 / (math.pi * e * AR)
CD_cruise = CD0 + K * CL_cruise**2
LD_cruise = CL_cruise / CD_cruise if CD_cruise > 0 else float("inf")

# Power (cruise and climb)
P_cruise = power_cruise_fuel(k=1, rho_alt=rho_cruise, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, n=n, b=b_cruise,
                                 MTOM=MTOM, S=S, V=V_cruise, eta_prop=eta_prop_cruise)
P_climb = power_climb_fuel(k=1, rho_alt=rho_climb, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, b=b_climb,
                                 MTOM=MTOM, S=S, V=V_climb, theta=theta_climb, eta_prop=eta_prop_cruise)
P_SL = max(P_cruise, P_climb)

# Power (hover)
P_hover = 1.15 * power_hover(MTOM=MTOM, rho=rho_hover, R_prop=R_rotor, num_rotors=num_rotors, eta_h=eta_hover)

# Fuel mass
mass_fuel_cruise = M_hydrogen_turboprop_cruise(rho_sl=rho_cruise, MTOM=MTOM, S=S, V=V_cruise, AR=AR, e=e, CD0=CD0,
                                                   b=b_cruise, R=range_cruise, eta_prop=eta_prop_cruise,
                                                   eta_elec=eta_fc, LHV=LHV)
mass_fuel = mass_fuel_cruise

# Energy
energy_battery = max(P_hover * t_hover, P_hover / c_rate)
energy_fuel = mass_fuel * rho_fuel
energy_trip = energy_battery + energy_fuel

# Time
time_cruise = (range_cruise / V_cruise) / 3600.0
time_climb = ((h_cruise - h_climb) / (V_climb * math.sin(math.radians(theta_climb)))) / 3600.0
time_trip = time_cruise + time_climb

# Mass breakdown
mass_motor_hover = M_motor(P_out=P_hover, rho_motor=rho_motor)
mass_motor_cruise = M_motor(P_out=P_SL, rho_motor=rho_motor)
mass_rotor = M_rotor(num_cruise=0, num_hover=num_rotors, R_rotor_cruise=R_rotor, R_rotor_hover=R_rotor, k_rotor=k_rotor)
mass_systems = M_systems(n=n, MTOM=MTOM, l_f=l_f, b=b_wing)
mass_wing = M_wing(b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio, t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_payload = M_payload(M_pax=m_pax, M_lug=m_lugg, n_seats=n_seats)
mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_furnish = M_furnish(MTOM=MTOM, n_crew=n_crew, V_cruise=V_cruise, b=b_wing, c=c)
mass_fuel_sys = mass_fuel / rho_fuel_sys
mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=e_usable)
mass_fuel_cell = M_fuel_cell(power_cruise=P_SL, rho_fuel_cell=rho_fuel_cell)

mass_total = (
        mass_motor_hover + mass_motor_cruise + mass_rotor + mass_systems + mass_wing + 
        mass_payload + mass_fuel_cell + mass_fuselage + mass_furnish + mass_fuel_sys + mass_battery
    )

print(f"Hover power: {P_hover}")
print(f"Cruise power: {P_cruise}")
print(f"Mass cruise motor: {mass_motor_cruise}")
print(f"Mass hover motor: {mass_motor_hover}")
print(f"Mass rotor: {mass_rotor}")
print(f"Mass systems: {mass_systems}")
print(f"Mass wing: {mass_wing}")
print(f"Mass fuselage: {mass_fuselage}")
print(f"Mass furnish: {mass_furnish}")
print(f"Mass fuel sys: {mass_fuel_sys}")
print(f"Mass battery: {mass_battery}")
print(f"Mass fuel cell: {mass_fuel_cell}")
print(f"Mass total: {mass_total}")
print(f"MTOM: {MTOM}")


