### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
import aerosandbox.numpy as np
from tabulate import tabulate

# Aircraft visualization imports
from DesignOS.EVTOLMDO.Optimization import create_aircraft_geometry

# Power model imports
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover

# Thrust model imports 
from DesignOS.EVTOLMDO.Model_functions.Thrust_model import thrust_required_cruise_asb
from DesignOS.EVTOLMDO.Model_functions.Thrust_model import thrust_required_climb_asb

# Mass model imports
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_rotor
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_motor
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_systems
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_payload
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_wing
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_fuselage
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_furnish
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_engine
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_fuel_jet_cruise_asb
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_battery


### --- Input Parameters --- ### 
rho_batt = 180.0 # energy density of the battery [Wh/kg]
c_rate = 10 
t_hover = 120 / 3600 # hover time [h]
n = 1 # ultimate load factor [-]
l_f = 6.0 # fuselage length [m]
w_f = 1.0 
d_f = 3.0 
num_rotors = 4 # number of rotors [-]
sweep = 0
taper_ratio = 1 
t_over_c = 0.12 # thickness to chord ratio [-]
m_pax = 82
n_seats = 4
m_lugg = 16
n_crew = 0 # number of crew members
T_to_W = 4 # engine thrust to weight ratio
range_cruise = 600 * 10**3 #cruise range [m]
TSFC = 1.4 * 10**-5 # thrust specific fuel consumption [kg/(N*s)]
rho_fuel_sys = 0.95 # fuel system mass density [-]
CD0 = 0.020
AR = 5
e = 0.85
CL_max = 1.5
eta_elec = 0.9
eta_prop_hover = 0.7
eta_hover = eta_elec * eta_prop_hover
rho_motor = 3225.806 # W/kg
k_rotor = 1.5 # rotor mass calculation correction factor
b_cruise = 0.95
b_climb = 1.0
h_cruise = 1000
h_climb = 100
theta_climb = 10
h_hover = 100
rho_cruise = float(np.squeeze(Atmosphere(h_cruise).density))
rho_climb = float(np.squeeze(Atmosphere(h_climb).density))
rho_hover = float(np.squeeze(Atmosphere(h_hover).density))
rho_sl = float(np.squeeze(Atmosphere(0).density))
g = 9.81
rho_fuel = 12000 # Wh/kg


### --- Optimization Parameters --- ###
opti = asb.Opti()

MTOM = opti.variable(init_guess=3000, log_transform=True, lower_bound=1, upper_bound=5700)
S = opti.variable(init_guess=50, log_transform=True, lower_bound=1, upper_bound=100)
V_cruise = opti.variable(init_guess=100, log_transform=True, lower_bound=1, upper_bound=300)
V_climb = opti.variable(init_guess=100, log_transform=True, lower_bound=1, upper_bound=300)
AR = opti.variable(init_guess=4, log_transform=True, lower_bound=1, upper_bound=5)
A_disk = opti.variable(init_guess=30, log_transform=True, lower_bound=1, upper_bound=100)
R_rotor = np.sqrt((A_disk/num_rotors)/np.pi) # rotor radius [m]


### --- Models --- ###
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

# Thrust processing 
T_SL_cruise = thrust_required_cruise_asb(k=1, rho_alt=rho_cruise, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                n=n, b=b_cruise, MTOM=MTOM, S=S, V=V_cruise)
T_SL_climb = thrust_required_climb_asb(k=1, rho_alt=rho_climb, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                b=b_climb, MTOM=MTOM, S=S, V=V_climb, theta=theta_climb)

# Power processing
P_hover = 1.15 * power_hover(MTOM=MTOM, rho=rho_hover, R_prop=R_rotor, num_rotors=num_rotors, eta_h=eta_hover)

# Fuel mass processing 
mass_fuel_cruise = M_fuel_jet_cruise_asb(TSFC=TSFC, rho_cruise=rho_cruise, MTOM=MTOM, S=S, V=V_cruise, AR=AR, e=e, CD0=CD0, b=b_cruise, R=range_cruise, g=g)
mass_fuel_climb = MTOM * 0.05
mass_fuel = mass_fuel_cruise + mass_fuel_climb

# Energy processing
energy_battery = P_hover/c_rate
energy_fuel = mass_fuel * rho_fuel 
energy_trip = energy_battery + energy_fuel

# Time processing
time_cruise = (range_cruise / V_cruise) / 3600 # hours
time_climb = ((h_cruise - h_climb) / (V_climb * math.sin(math.radians(theta_climb)))) / 3600 # hours
time_trip = time_cruise + time_climb

# Mass processing
mass_motor = M_motor(P_out=P_hover, rho_motor=rho_motor)
mass_rotor = M_rotor(num_cruise=0, num_hover=num_rotors, R_rotor_cruise=R_rotor, R_rotor_hover=R_rotor, k_rotor=k_rotor)
mass_systems = M_systems(n=n, MTOM=MTOM, l_f=l_f, b=b_wing)
mass_wing = M_wing(b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio, t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_payload = M_payload(M_pax=m_pax, M_lug=m_lugg, n_seats=n_seats)
mass_engine = M_engine(T_to_W=T_to_W, T_cruise=T_SL_climb)
mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_furnish = M_furnish(MTOM=MTOM, n_crew=n_crew, V_cruise=V_cruise, b=b_wing, c=c)
mass_fuel_sys = mass_fuel / rho_fuel_sys
mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=0.80)
mass_total = mass_motor + mass_rotor + mass_systems + mass_wing + mass_payload + mass_engine + mass_fuselage + mass_furnish + mass_fuel_sys + mass_battery


### --- Constraints --- ###
opti.subject_to([
    MTOM <= 5700,
    MTOM == mass_total,
    V_cruise > 120,
    V_climb > 100,
    b_wing <= 15,
    A_disk / S <= 0.45
])


### --- Optimization objective --- ###
opti.minimize(energy_trip) 


### --- Solve ---
sol = opti.solve(max_iter=300)


### --- Display Results --- ###
print("--- Optimization Variables ---")
print(f"Optimal V_cruise: {sol.value(V_cruise * 3.6)} kph")
print(f"Opimal V_climb: {sol.value(V_climb * 3.6)} kph")
print(f"Optimal MTOM: {sol.value(MTOM)} kg")
print(f"Optimal Disk Area: {sol.value(A_disk)} m^2")
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
print(f"Thrust available at sea level for cruise: {sol.value(T_SL_cruise)} N")
print(f"Thrust available at sea level for climb: {sol.value(T_SL_climb)} N")
print(f"Thrust available at sea level: {sol.value(T_SL_climb)} N")
print(f"Thrust-to-weight ratio: {sol.value(T_SL_climb / (MTOM * g))}")
print(f"Total hover power required: {sol.value(P_hover / 1000)} kW")
print(" ")
print("--- Energy Properties ---")
print(f"Battery energy required: {sol.value(energy_battery / 1000)} kWh")
print(f"Fuel energy required: {sol.value(energy_fuel / 1000)} kWh")
print(f"Total trip energy required: {sol.value(energy_trip / 1000)} kWh")
print(" ")
print("--- Time Properties ---")
print(f"Total cruise time: {sol.value(time_cruise)} hours")
print(f"Total climb time: {sol.value(time_climb)} hours")
print(f"Total trip time: {sol.value(time_trip)} hours")
print(" ")
print("--- Mass Properties ---")
print(f"Total motor mass: {sol.value(mass_motor)} kg")
print(f"Total rotor mass: {sol.value(mass_rotor)} kg")
print(f"Total systems mass: {sol.value(mass_systems)} kg")
print(f"Total payload mass: {sol.value(mass_payload)} kg")
print(f"Total wing mass: {sol.value(mass_wing)} kg")
print(f"Total fuselage mass: {sol.value(mass_fuselage)} kg")
print(f"Total furnishing mass: {sol.value(mass_furnish)} kg")
print(f"Total battery mass: {sol.value(mass_battery)} kg")
print(f"Total engine mass: {sol.value(mass_engine)} kg")
print(f"Total fuel mass: {sol.value(mass_fuel)} kg")
print(f"Maximum takeoff mass: {sol.value(mass_total)} kg")
