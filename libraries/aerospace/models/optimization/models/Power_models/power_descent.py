import math 

def power_descent(MTOM:float, rho:float, c:float, b:float,
                  V_descent:float, ROD:float, C_D:float,
                  R_prop:float, num_props:float, eta_c:float, P_min:float) -> float:
    
    """ Computes power required by cruise ESS during fixed wing descent

    Inputs:
    MTOM -> maximum take-off mass [kg]
    rho -> air density [kg/m^3]
    c -> chord length [m]
    b -> wing span [m]
    V_descent -> descent speed [m/s]
    ROD -> rate of descent [m]
    C_D -> drag coefficient [-]
    R_prop -> cruise propeller radius [m]
    num_props -> number of props for cruise [-]
    eta_c -> cruise engine propulsive efficiency [-]
    P_min -> minimum power required for stabilization during descent [W]
    
    Outputs:
    power_descent -> power required during descent [W]

    """
    g = 9.81 # earth acceleration in m/s^2
    W = MTOM * g # aircraft weight in N
    S = c * b # wing area in m^2 
    V_descent_calc = math.sqrt(V_descent**2 + ROD**2) # total descent velocity in m/s
    D = 0.5 * rho * (V_descent_calc**2) * S * C_D # drag in descent N
    T_required = D - (W * ROD / V_descent_calc) # thrust required in descent in N
    
    # Ensure T_required is non-negative
    if T_required < 0:
        T_required = 0

    A_prop = math.pi * R_prop**2 # propeller disk area in m^2 
    T_prop = T_required / num_props # thrust per propeller in N
    sigma = T_prop / A_prop # disk loading in N/m^2 
    vi = math.sqrt(T_prop / (2 * rho * A_prop)) # induced velocity in m/s

    # if propeller powered
    # power_descent_calc = (T_required * V_descent_calc) / eta_c + (T_prop * vi) / eta_c * num_props # Power required for descent (P_descent) including induced power
    power_descent_calc = (T_required * V_descent_calc) / eta_c
    
    power_descent = max(power_descent, P_min) # total power in descent, considering total minimum power required

    return power_descent


