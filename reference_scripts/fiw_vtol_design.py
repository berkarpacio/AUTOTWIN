### --- Imports ---
# Generic imports
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
from tabulate import tabulate

# Aircraft visualization imports
from DesignOS.EVTOLMDO.Optimization import create_aircraft_geometry

# Power model imports
from DesignOS.EVTOLMDO.Model_functions.Power_model import power_hover

# Thrust model imports 
from DesignOS.EVTOLMDO.Model_functions.Thrust_model import thrust_required_cruise
from DesignOS.EVTOLMDO.Model_functions.Thrust_model import thrust_required_climb

# Mass model imports
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_rotor
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_motor
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_systems
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_payload
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_wing
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_fuselage
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_furnish
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_engine
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_fuel_jet_cruise
from DesignOS.EVTOLMDO.Model_functions.Weight_model import M_battery

### --- Aircraft Parameters ---
# Aerodynamic parameters
CD0 = 0.020
AR = 5
e = 0.85
CL_max = 1.5
MTOM = 4000 # determined after constraint analysis
S = 65 # determined after constraint analysis

# Hover mission parameters
alt_hover = 100
rho_hover = float(np.squeeze(Atmosphere(alt_hover).density))
eta_elec = 0.9
eta_prop_hover = 0.7
eta_hover = eta_elec * eta_prop_hover
num_rotors = 1
rho_motor = 3225.806 # W/kg

# Cruise Mission parameters
alt_cruise = 1000
alt_sl = 0
rho_sl = float(np.squeeze(Atmosphere(alt_sl).density))
rho_cruise = float(np.squeeze(Atmosphere(alt_cruise).density))
V_cruise = 500 * 0.277777778
g = 9.81
k_cruise = 1
n = 1
b_cruise = 0.95

# Climb Mission parameters
alt_climb = 100
alt_sl = 0
rho_sl = float(np.squeeze(Atmosphere(alt_sl).density))
rho_climb = float(np.squeeze(Atmosphere(alt_cruise).density))
V_climb = 500 * 0.277777778
k_climb = 1
b_climb = 1
theta_climb = 10

# Stall Speed Limit
V_stall = math.sqrt((2*MTOM*g)/(S*rho_sl*CL_max))
rho_stall = float(np.squeeze(Atmosphere(alt_cruise).density))
wing_loading_stall = 0.5 * rho_stall * CL_max * V_stall**2

### --- Constraint Analysis --- 
# Thrust Loading vs. Wing Loading
m = np.arange(10, 5700, 1)
S = np.linspace(1, 40, num=m.size)
wing_loading = (m*g)/S

T_SL_cruise = thrust_required_cruise(k=k_cruise, rho_alt=rho_cruise, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                n=n, b=b_cruise, MTOM=m, S=S, V=V_cruise)
thrust_loading_cruise = T_SL_cruise/(m*g)

T_SL_climb = thrust_required_climb(k=k_climb, rho_alt=rho_climb, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                b=b_climb, MTOM=m, S=S, V=V_climb, theta=theta_climb)
thrust_loading_climb = T_SL_climb/(m*g)

plt.figure()
plt.plot(wing_loading, thrust_loading_cruise, label="Constant Speed Cruise")
plt.plot(wing_loading, thrust_loading_climb, label="Constant Speed Climb")
plt.axvline(x=wing_loading_stall, color="r", linestyle="--", label="Stall Limit")
plt.title("Thrust Loading vs. Wing Loading", fontsize=18)
plt.xlabel("Wing Loading (N/m^2)", fontsize=14)
plt.ylabel("Thrust Loading (N/N)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

# Hover Power Required vs. Disk Area
A_vector = np.arange(1, 50, 1) # total rotor area [m^2]
P_hover_vector = []

for A in A_vector:
    R_rotor = math.sqrt(A/(math.pi))
    # Excess power 1.15 * hover_power 
    hover_power = 1.15 * power_hover(MTOM=4000, rho=rho_hover, R_prop=R_rotor, num_rotors=num_rotors, eta_h=eta_hover)
    P_hover_vector.append(hover_power/1000)

plt.figure()
plt.plot(A_vector, P_hover_vector)
plt.title("Hover Power Required vs. Total Disk Area", fontsize=18)
plt.xlabel("Total Disk Area (m^2)", fontsize=14)
plt.ylabel("Hover Power Required (kW)", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()


### --- Design Point Analysis ---
# Configurational data
MTOM = 4000 # maximum takeoff mass [kg]
rho_batt = 180.0 # energy density of the battery [Wh/kg]
c_rate = 10 
t_hover = 120 / 3600 # hover time [h]
S = 65.333 # main wing area [m^2]
b_wing = (S * AR) ** 0.5 # wing span [m]
c = S/b_wing # wing mac [m]
n = 1 # ultimate load factor [-]
l_f = 6.0 # fuselage length [m]
w_f = 1.0 
d_f = 3.0 
A_total = 30 # total disk area [m^2]
num_rotors = 4 # number of rotors [-]
R_rotor = math.sqrt((A_total/num_rotors)/math.pi) # rotor radius [m]
sweep = 0
taper_ratio = 1 
t_over_c = 0.12 # thickness to chord ratio [-]
m_pax = 82
n_seats = 4
m_lugg = 16

n_crew = 0 # number of crew members
T_to_W = 4 # engine thrust to weight ratio
range_cruise = 500 * 10**3 # cruise range [m]
TSFC = 1.4 * 10**-6 # thrust specific fuel consumption [kg/(N*s)]
rho_fuel_sys = 0.95 # fuel system mass density [-]

# Aerodynamic processing
# Drag polar
V_vector = np.arange(V_stall, 200 ,1 )
print(b_cruise)
print(MTOM)
print(g)
print(rho_cruise)
print(S)
CL_vector_alt = (2*b_cruise*MTOM*g)/(rho_cruise*S*V_vector**2)
CD_vector_alt = CD0 + (1/(math.pi*e*AR)) * CL_vector_alt**2
Drag_alt = 0.5 * rho_cruise * S * CD_vector_alt * V_vector**2
CL_vector_sl = (2*b_cruise*MTOM*g)/(rho_sl*S*V_vector**2)
CD_vector_sl = CD0 + (1/(math.pi*e*AR)) * CL_vector_sl**2
Drag_sl = 0.5 * rho_sl * S * CD_vector_sl * V_vector**2
CL_over_CD_alt = CL_vector_alt / CD_vector_alt
plt.figure()
plt.plot(CD_vector_alt, CL_vector_alt)
plt.title("Drag Polar", fontsize=18)
plt.xlabel("CD", fontsize=14)
plt.ylabel("CL", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

# Drag vs. speed and altitude
plt.figure()
plt.plot(V_vector, Drag_alt, label="Drag at altitude")
plt.plot(V_vector, Drag_sl, label="Drag at sl")
plt.title("Variation of drag with speed", fontsize=18)
plt.xlabel("V (m/s)", fontsize=14)
plt.ylabel("Drag (N)", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()

# Power and thrust processing 
hover_power_required = 1.15 * power_hover(MTOM=MTOM, rho=rho_hover, R_prop=R_rotor, num_rotors=num_rotors, eta_h=eta_hover)
cruise_thrust_required = thrust_required_cruise(k=k_cruise, rho_alt=rho_cruise, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                n=n, b=b_cruise, MTOM=MTOM, S=S, V=V_cruise)
climb_thrust_required = thrust_required_climb(k=k_climb, rho_alt=rho_climb, rho_sl=rho_sl, CD0=CD0, AR=AR, e=e, 
                                                b=b_climb, MTOM=MTOM, S=S, V=V_climb, theta=theta_climb)

# Energy Processing 
energy_mission = hover_power_required * t_hover
energy_battery = max(energy_mission, hover_power_required/c_rate)
print(hover_power_required)
print(energy_battery)

# Mass Processing
mass_motor = M_motor(P_out=hover_power_required, rho_motor=rho_motor)
mass_rotor = M_rotor(num_cruise=0, num_hover=num_rotors, R_rotor_cruise=R_rotor, R_rotor_hover=R_rotor)
mass_systems = M_systems(n=n, MTOM=MTOM, l_f=l_f, b=b_wing)
mass_wing = M_wing(b=b_wing, c=c, sweep=sweep, taper_ratio=taper_ratio, t_over_c=t_over_c, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_payload = M_payload(M_pax=m_pax, M_lug=m_lugg, n_seats=n_seats)
mass_engine = M_engine(T_to_W=T_to_W, T_cruise=MTOM*g*1.2)
mass_fuselage = M_fuselage(l_f=l_f, w_f=w_f, d_f=d_f, n=n, MTOM=MTOM, V_cruise=V_cruise)
mass_furnish = M_furnish(MTOM=MTOM, n_crew=n_crew, V_cruise=V_cruise, b=b_wing, c=c)
mass_fuel_cruise = M_fuel_jet_cruise(TSFC=TSFC, rho_cruise=rho_cruise, MTOM=MTOM, S=S, V=V_cruise, AR=AR, e=e, CD0=CD0, b=b_cruise, R=range_cruise, 
                              rho_fuel_sys=rho_fuel_sys, g=g)
mass_fuel_climb = MTOM * 0.05
mass_fuel = mass_fuel_cruise + mass_fuel_climb
mass_battery = M_battery(E_trip=energy_battery, E_reserve=0, rho_bat=rho_batt, e_usable=0.80)
mass_total = mass_motor + mass_rotor + mass_systems + mass_wing + mass_payload + mass_engine + mass_fuselage + mass_furnish + mass_fuel + mass_battery
print(mass_total)

### --- Draw Aircraft ---
S_wing = S
fuselage_l = l_f
AR_wing = AR
taper_wing=0.4
theta_wing=30.0*(math.pi/180)
c_HT=0.3
tail_arm_perc=0.9
AR_hstab=4
taper_hstab=0.6
c_VT=0.0499
AR_vstab=2.27 
taper_vstab=0.5

output = create_aircraft_geometry(
        S_wing = S_wing, 
        AR_wing = AR_wing, 
        taper_wing =  taper_wing, 
        theta_wing = theta_wing, 
        c_HT=c_HT, 
        tail_arm_perc=tail_arm_perc, 
        AR_hstab=AR_hstab, 
        taper_hstab=taper_hstab, 
        c_VT=c_VT, 
        AR_vstab=AR_vstab, 
        taper_vstab=taper_vstab,
        fuselage_l=fuselage_l
    )    

airplane = output[0]
wing_params = output[1]
hstab_params = output[2]
vstab_params = output[3]

def print_table(d: dict, title: str):
    print(f"\n{title}")
    print(tabulate([(k, v) for k, v in d.items()],
                   headers=["Parameter", "Value"],
                   tablefmt="github"))

print_table(wing_params, "Main Wing Parameters")
print_table(hstab_params, "Horizontal Stabilizer Parameters")
print_table(vstab_params, "Vertical Stabilizer Parameters")
airplane.draw_three_view()
