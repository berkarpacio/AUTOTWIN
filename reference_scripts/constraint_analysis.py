### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt


### --- Hover Mission ---
# Mission Parameters
MTOM_values = [1000, 2000, 3000, 4000, 5000] # aircraft maximum takeoff mass [kg]
g = 9.81 # acceleration due to gravity [m/s^2]
V_i = 0 * 0.278 # aircraft initial speed [m/s]
V_f = 9 * 0.278 # aircraft target vertical climb speed [m/s]
delta_t = 120 # time required to reach the target climb speed [s]
k = 1.15 # induced power factor [-]
sigma_solidity = 0.20 # lift fan solidity
cd0 = 0.008 # profile drag coefficient of the lift fan blades [-]
cl = 0.7 # lift coefficient of the lift fan blades [-]
alt = 15
rho = float(np.squeeze(Atmosphere(alt).density))
eta_h = 0.7

# Initialize
A_vector = np.arange(1, 50, 1) # total fan area [m^2]

plt.figure()
colors = plt.cm.viridis(np.linspace(0, 1, len(MTOM_values)))
for idx, MTOM in enumerate(MTOM_values):
    P_vector_fiw = []
    P_vector_mt = []

    for A in A_vector:
        power_fiw = (((MTOM*g)**1.5)/(math.sqrt(rho))) * (k/math.sqrt(2) + (cd0/(8*math.sqrt(sigma_solidity))) * (6/cl)**1.5) * (1/math.sqrt(A))
        sigma = (MTOM*g) / A
        power_mt = (MTOM * g / eta_h) * np.sqrt(sigma / (2 * rho))
        P_vector_fiw.append(power_fiw/1000)
        P_vector_mt.append(power_mt/1000)

    color = colors[idx]
    plt.plot(A_vector, P_vector_fiw, color=color, linestyle="-", label=f"MTOM={MTOM} kg - FIW")
    plt.plot(A_vector, P_vector_mt, color=color, linestyle="--", label=f"MTOM={MTOM} kg - MT")

plt.title("Hover Power Required vs. Total Disk Area", fontsize=18)
plt.xlabel("Total Disk Area (m^2)", fontsize=14)
plt.ylabel("Hover Power Required (kW)", fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()


### --- Cruise Mission ---
# Mission Parameters
alt = 500
rho_sl = float(np.squeeze(Atmosphere(0).density))
rho = float(np.squeeze(Atmosphere(alt).density))
sigma = rho/rho_sl
k = 0.8
CD0 = 0.020
AR = 5
e = 0.80
n = 1
b = 0.95
V_cruise = 700 * 0.277777778

# Initialize vectors
m = np.arange(10, 5700, 1)
S = np.linspace(1, 40, num=m.size)
wing_loading_cruise = (m*g)/S

T_SL_cruise = (1/(2*k*sigma)) * (CD0 + (1/(math.pi*AR*e)) * (2*n*b*m*g/(rho*S*V_cruise**2))**2) * rho*S*V_cruise**2
thrust_loading_cruise = T_SL_cruise/(m*g)

### --- Climb Mission ---
# Mission Parameters
alt = 100
rho_sl = float(np.squeeze(Atmosphere(0).density))
rho = float(np.squeeze(Atmosphere(alt).density))
sigma = rho/rho_sl
k = 1 
CD0 = 0.020
AR = 5
e = 0.80
n = 1
b = 1
V_climb = 700 * 0.277777778
theta = math.radians(10)

# Initialize vectors
m = np.arange(10, 5700, 1)
S = np.linspace(1, 40, num=m.size)
wing_loading_climb = (m*g)/S

T_SL_climb = (1/(k*sigma)) * ((1/2) * (CD0 + (1/(math.pi*AR*e)) * (2*n*b*m*g*math.cos(theta)/(rho*S*V_cruise**2))**2) * rho*S*V_cruise**2 + b*m*g*math.sin(theta))
thrust_loading_climb = T_SL_climb/(m*g)

### --- Banked Turn Mission ---
# Mission Parameters
alt = 500
rho_sl = float(np.squeeze(Atmosphere(0).density))
rho = float(np.squeeze(Atmosphere(alt).density))
sigma = rho/rho_sl
k = 0.8
CD0 = 0.020
AR = 5
e = 0.80
bank = math.radians(50)
n = 1/math.cos(bank)
b = 1
V_turn = 700 * 0.277777778

# Initialize vectors
m = np.arange(10, 5700, 1)
S = np.linspace(1, 40, num=m.size)
wing_loading_bank = (m*g)/S

T_SL_bank = (1/(2*k*sigma)) * (CD0 + (1/(math.pi*AR*e)) * (2*n*b*m*g/(rho*S*V_turn**2))**2) * rho*S*V_turn**2
thrust_loading_bank = T_SL_bank/(m*g)

### --- Stall Speed Limit ---
V_stall = 100 * 0.277777778
alt = 100
rho = float(np.squeeze(Atmosphere(alt).density))
CL_max = 1.5
wing_loading_stall = 0.5 * rho * CL_max * V_stall**2

### --- Required Fuel Weight for Cruise ---
R = 2000 * 10**3 # range [m]
TSFC = 2.0 * 10**-5 # thrust-specific fuel consumption [kg/(N*s)]
m = 4000
alt = 500
rho = float(np.squeeze(Atmosphere(alt).density))
K = (1/(math.pi*AR*e))
b = 0.95
S = 60
rho_fuel_sys = 0.06 # fuel storage system mass density
CL_cruise = (2*b*m*g)/(rho*S*V_cruise**2)
CD_cruise = CD0 + K*CL_cruise**2
ff = math.exp(R/((1/TSFC*g) * math.sqrt(2*b*m*g/(rho*S)) * CL_cruise**0.5/(CD_cruise)))
m_fuel = (ff*m*b - m*b)/ff
m_fuel_sys = m_fuel / rho_fuel_sys

### --- Required Fuel Weight for Climb ---


### --- Required Battery Weight for Hovering ---

plt.figure()
plt.plot(wing_loading_cruise, thrust_loading_cruise, label="Constant Speed Cruise")
plt.plot(wing_loading_climb, thrust_loading_climb, label="Constant Speed Climb")
plt.plot(wing_loading_bank, thrust_loading_bank, label="Constant Speed Bank")
plt.axvline(x=wing_loading_stall, color="r", linestyle="--", label="Stall Limit")
plt.title("Thrust Loading vs. Wing Loading", fontsize=18)
plt.xlabel("Wing Loading (N/m^2)", fontsize=14)
plt.ylabel("Thrust Loading (N/N)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()
print(m_fuel_sys)
