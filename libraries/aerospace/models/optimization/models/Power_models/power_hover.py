import math

def power_hover(MTOM:float, rho:float, R_prop:float, num_rotors:float, eta_h:float) -> float:
    
    """ Computes power required by hover ESS during hover

    Inputs:
    MTOM -> maximum take-off mass [kg]
    rho -> air density [kg/m^3]
    R_prop -> hover propeller radius [m]
    num_rotors -> number of rotors for hover [-]
    eta_h -> hover propulsor efficiency [-]
    
    Outputs:
    power_hover -> power required during hover [W]

    """ 
    g = 9.81 # earth acceleration in m/s^2
    T_total = MTOM * g # total thrust required for hover in N
    T_rotor = T_total / num_rotors # thrust per rotor in N
    A_rotor = math.pi * R_prop**2 # single rotor disk area in m^2 
    A_total = num_rotors * A_rotor # total rotor disk area in m^2 
    sigma = T_total / A_total # total disk loading in N/m^2 
    vi = math.sqrt(T_rotor / (2 * rho * A_rotor)) # induced velocity in m/s
    P_i = (T_total * vi) / eta_h # induced power in W
    power_hover = (MTOM * g / eta_h) * math.sqrt(sigma / (2 * rho)) # power equation in W

    return power_hover
