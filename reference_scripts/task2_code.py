### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt


### --- Aircraft Data ---
### Aerodynamic Data
CL_0 = 0.4 # Lift coefficient of aircraft at zero alpha [-]
CL_alpha = 5.7 # Lift coefficient vs. alpha curve slope [1/rad]
CL_at_CD_min = 0.15 # Lift coefficient of aircraft at minimum drag coefficient [-]
CD_min = 0.022 # Parasitic drag coefficient [-]
K = 0.035 # Induced drag coefficient [-]
alpha_stall = 15 # Angle of attack at stall [deg]
S_ref = 20 # Wing reference area [m^2]
g = 9.81 # Acceleration due to gravity [m/s^2]

### Mass Data
MTOM = 2500 # Maximum take-off mass [kg]
MTOW = 2500 * g # Maximum take-off weight [N]
V_fuel_max = 1200 # Maximum fuel volume [L]
rho_fuel = 800 # Fuel density [kg/m^3]
fuel_remaining = 200 # Volume of fuel remaining at the beginning of descent [L]

V_fuel_max_m3 = V_fuel_max * 1e-3 # Convert to SI [m^3]
V_fuel_final_m3 = fuel_remaining * 1e-3 # Convert to SI [m^3] 

m_fuel_max = V_fuel_max_m3 * rho_fuel # Starting fuel mass [kg]
m_fuel_final = V_fuel_final_m3 * rho_fuel # Remaining fuel mass [kg]

m_initial = MTOM
m_empty = MTOM - m_fuel_max
m_final_target = MTOM - (m_fuel_max-m_fuel_final)

### Propulsion data
A_prop = 1.9 # Propeller area [m^2]
eta_prop = 0.8 # Propeller efficiency [-]
P_TO_W = 350 * 745.7 # Take-off power [W]
SFC_TO_SI = 0.28 / (3600 * 1000) # Specific fuel consumption at take off power [kg/W/s]
P_max_cont_W = 550 * 745.7 # Maximum continuous power [W]
SFC_max_cont_SI = 0.25 / (3600 * 1000) # Specific fuel consumption at max continuous power [kg/W/s]
P_cruise_W = 280 * 745.7 # Cruise power [W]
SFC_cruise_SI = 0.2 / (3600 * 1000) # Specific fuel consumption at cruise power [kg/W/s]

### Mission data
alt_0 = 20000 * 0.3048 # Cruise start altitude [m]


### Define aerodynamic helper functions
def CL_from_alpha(alpha_deg: float) -> float:
    alpha_rad = math.radians(alpha_deg)
    CL = CL_0 + CL_alpha * alpha_rad
    return CL

def CD_from_CL(CL:float) -> float:
    CD = CD_min + K * (CL - CL_at_CD_min)**2
    return CD

### Define mission simulation 
def simulate_mission(dt=5):

    # Initialize aircraft states
    t = 0.0 # time [s]
    x = 0.0 # ground distance [m]
    h = 0.0 # altitude [m]
    y = 0.0 # y distance [m]
    y_dot = 0.0 # y velocity [m/s]
    m_fuel = m_fuel_max # starting fuel mass [kg]
    m = MTOM # starting aircraft mass [kg]
    V = 5 # initial speed of aircraft [m/s]
    throttle = 0.01 # initial throttle [-]
    throttle_take_off = 1.0 # max throttle during take-off [-]
    throttle_rate = 0.05 # throttle ramp rate [s^-1]
    climb_alt = 20000 * 0.3048 # climb altitude [m]


    # Mission segment bookkeeping
    segment = 1  # 1...7
    
    # Histories
    t_hist = []
    h_hist = []
    x_hist = []
    y_hist = []
    m_hist = []
    m_fuel_hist = []
    segment_hist = []
    V_hist = []
    throttle_hist = []

    # Mission loop
    while m_fuel > 0 and segment != 8:

        rho = float(np.squeeze(Atmosphere(h).density)) # air density update update using the standard atmosphere model [kg/m^3]
        rho_sl = float(np.squeeze(Atmosphere(0).density)) # air density at sea level [kg/m^3]
        sigma = rho/rho_sl # density ratio [-]
        W = m * g # aircraft weight [N]
        y_dot = 0.0

        # Take-off segment
        if segment == 1:

            # Take-off settings
            alpha_takeoff = 10
            CL_max = CL_from_alpha(alpha_stall)
            V_stall = math.sqrt(2*W/(rho*S_ref*CL_max))
            CL_takeoff = CL_from_alpha(alpha_takeoff)
            CD_takeoff = CD_from_CL(CL_takeoff)
            throttle = min(throttle + throttle_rate * dt, throttle_take_off)

            # Take-off physics
            P_shaft = (P_TO_W) * throttle
            m_dot_fuel = SFC_TO_SI * P_shaft
            T = (P_shaft * eta_prop) / V
            D = 0.5 * rho * CD_takeoff * S_ref * V**2
            L = 0.5 * rho * CL_takeoff * S_ref * V**2
            V_dot = (T - D) / m
            x_dot = V
            h_dot = 0

            # Segment switching logic
            if L > W and V >= 1.2 * V_stall:
                segment = 2

        # Constant TAS climb segment
        elif segment == 2: 

            # Climb settings
            P_max = P_max_cont_W
            SFC = SFC_max_cont_SI
            V_climb = V_stall * 1.2 # climb speed [m/s]
            V = V_climb
            climb_angle = math.radians(5)

            # Climb physics
            L = W * math.cos(climb_angle)
            CL_climb = L / (0.5 * rho * S_ref * V_climb**2)
            CD_climb = CD_from_CL(CL_climb)

            D = 0.5 * CD_climb * rho * S_ref * V_climb**2
            T = D + W * math.sin(climb_angle)

            x_dot = V * (L/W)
            h_dot = V * (T-D)/W
            V_dot = 0

            P_shaft = (T * V) / eta_prop
            m_dot_fuel = P_shaft * SFC
            throttle = P_shaft/(P_max * sigma)

            # Segment switching logic
            if h >= climb_alt:
                segment = 3
        
        # Steady-level cruise segment
        elif segment == 3:

            # Cruise settings
            P_max = P_cruise_W
            V_cruise = 80
            V = V_cruise
            SFC = SFC_cruise_SI

            # Cruise physics
            CL_cruise = (2*W)/(rho * S_ref * V**2)
            CD_cruise = CD_from_CL(CL_cruise)
            D = 0.5 * rho * S_ref * V**2 * CD_cruise
            T = D
            P_shaft = (T * V) / eta_prop
            throttle = P_shaft/P_max
            m_dot_fuel = SFC_cruise_SI * P_shaft
            x_dot = V
            h_dot = 0
            V_dot = 0

            # Segment switching logic
            if x >= 1000000:
                segment = 4
        
        # Constant airspeed and altitude loiter
        elif segment == 4:

            # Loiter settings
            phi = math.radians(10) # bank angle
            V_loiter = 60
            V = V_loiter
            n = 1 / math.cos(phi) # load factor
            turn_radius = V**2/(g*math.tan(phi))
            omega = V/turn_radius # angular speed
            psi = 0 # heading angle
            P_max = P_cruise_W
            SFC = SFC_cruise_SI

            while psi <  2*math.pi:
                
                CL_loiter = (2*W*n)/(rho * S_ref * V**2)
                CD_loiter = CD_from_CL(CL_loiter)
                D = 0.5 * rho * S_ref * V**2 * CD_loiter
                T = D
                P_shaft = (T * V) / eta_prop
                throttle = P_shaft/P_max
                m_dot_fuel = SFC * P_shaft
                
                x_dot = V * math.cos(psi)
                y_dot = V * math.sin(psi)
                h_dot = 0.0
                V_dot = 0.0

                # Integrate states
                V += V_dot * dt
                x += x_dot * dt
                y += y_dot * dt
                h += h_dot * dt
                t += dt
                m_fuel = max(m_fuel - m_dot_fuel * dt, 0.0)
                m = m_empty + m_fuel
                psi += omega * dt

                # Record history
                t_hist.append(t)
                h_hist.append(h)
                x_hist.append(x)
                y_hist.append(y)
                V_hist.append(V)
                m_hist.append(m)
                throttle_hist.append(throttle)
                m_fuel_hist.append(m_fuel)
                segment_hist.append(segment)

            # Segment switching logic, a full circle is completed
            segment = 5
            continue
        
        # Second steady-level cruise segment
        elif segment == 5:

            # Cruise settings
            P_max = P_cruise_W
            V_cruise = 80
            V = V_cruise
            SFC = SFC_cruise_SI

            # Cruise physics
            CL_cruise = (2*W)/(rho * S_ref * V**2)
            CD_cruise = CD_from_CL(CL_cruise)
            D = 0.5 * rho * S_ref * V**2 * CD_cruise
            T = D
            P_shaft = (T * V) / eta_prop
            throttle = P_shaft/P_max
            m_dot_fuel = SFC_cruise_SI * P_shaft
            x_dot = V
            h_dot = 0
            V_dot = 0

            # Segment switching logic
            if m_fuel <= m_fuel_final:
                segment = 6

        # Powered descent segment
        elif segment == 6:

            # Descent settings
            P_max = P_cruise_W
            SFC = SFC_cruise_SI
            V_descent = 20.0  # m/s
            V = V_descent
            gamma = math.radians(5) 

            # Descent physics
            L_req = W * math.cos(gamma)
            CL_descent = L_req / (0.5 * rho * S_ref * V_descent**2)
            CD_descent = CD_from_CL(CL_descent)

            D = 0.5 * CD_descent * rho * S_ref * V_descent**2
            T_req = D - W * math.sin(gamma) 

            # Kinematics
            x_dot = V * (L_req/W)
            h_dot = (V * (T_req - D)) / W 
            V_dot = 0

            # Propulsion
            P_shaft = (T_req * V) / eta_prop
            m_dot_fuel = P_shaft * SFC
            throttle = P_shaft / (P_max * sigma)

            # Segment switching logic
            if h <= 0:
                segment = 7

        # Landing
        elif segment == 7:

            # Landing settings
            alpha_landing = 10
            CL_max = CL_from_alpha(alpha_stall)
            CL_landing = CL_from_alpha(alpha_landing)
            CD_landing = CD_from_CL(CL_landing)
            throttle = 0
            mu_roll = 0.2  # simple braking coefficient

            # Landing physics
            P_shaft = (P_TO_W) * throttle
            m_dot_fuel = SFC_TO_SI * P_shaft
            T = 0.0
            D = 0.5 * rho * CD_landing * S_ref * V**2
            L = 0.5 * rho * CL_landing * S_ref * V**2

            # Friction proportional to normal force; clamp V >= 0
            V_dot = ((T - D) - mu_roll * (W - L)) / m
            x_dot = max(V, 0.0)
            h_dot = 0.0

            # Segment switching logic
            if V <= 0.5:
                segment = 8

        # Advance states
        V += V_dot * dt
        x += x_dot * dt
        y += y_dot * dt
        h += h_dot * dt
        t += dt
        m_fuel = max(m_fuel - m_dot_fuel * dt, 0.0)
        m = m_empty + m_fuel

        # Record current state
        t_hist.append(t)
        h_hist.append(h)
        y_hist.append(y)
        x_hist.append(x)
        V_hist.append(V)
        m_hist.append(m)
        throttle_hist.append(throttle)
        m_fuel_hist.append(m_fuel)
        segment_hist.append(segment)

    return {
        "t": t_hist,
        "h": h_hist,
        "y": y_hist,
        "m": m_hist,
        "m_fuel": m_fuel_hist,
        "V": V_hist,
        "throttle": throttle_hist,
        "segment": segment_hist,
        "x": x_hist
    }

if __name__ == "__main__":

    results = simulate_mission()
    t_hist = np.array(results["t"]) / 3600.0  # hours
    h_hist = np.array(results["h"]) / 1000.0  # km
    m_hist = np.array(results["m"])
    m_fuel_hist = np.array(results["m_fuel"])
    throttle_hist = np.array(results["throttle"])
    V_hist = np.array(results["V"])
    x_hist = np.array(results["x"]) / 1000.0  # km
    y_hist = np.array(results["y"]) / 1000.0  # km
    segment_hist = np.array(results["segment"])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes[0, 0].plot(t_hist, V_hist)
    axes[0, 0].set_ylabel("V [m/s]")
    axes[0, 0].set_xlabel("Time [hr]")
    axes[0, 0].set_title("Airspeed vs. Time")

    axes[0, 1].plot(t_hist, x_hist)
    axes[0, 1].set_ylabel("x [km]")
    axes[0, 1].set_xlabel("Time [hr]")
    axes[0, 1].set_title("x vs. Time")

    axes[1, 0].plot(t_hist, h_hist)
    axes[1, 0].set_ylabel("h [km]")
    axes[1, 0].set_xlabel("Time [hr]")
    axes[1, 0].set_title("Altitude vs. Time")

    axes[1, 1].plot(t_hist, m_hist)
    axes[1, 1].set_ylabel("Mass [kg]")
    axes[1, 1].set_xlabel("Time [hr]")
    axes[1, 1].set_title("Aircraft Mass vs. Time")

    axes[2, 0].plot(t_hist, y_hist)
    axes[2, 0].set_ylabel("y [km]")
    axes[2, 0].set_xlabel("Time [hr]")
    axes[2, 0].set_title("y vs. Time")

    axes[2, 1].plot(t_hist, throttle_hist)
    axes[2, 1].set_ylabel("Throttle [-]")
    axes[2, 1].set_xlabel("Time [hr]")
    axes[2, 1].set_title("Throttle vs. Time")

    fig.suptitle("Mission Simulation Results", fontsize=14)
    fig.tight_layout(h_pad=2.0, w_pad=1.5, rect=(0, 0, 1, 0.97))
    plt.show()

