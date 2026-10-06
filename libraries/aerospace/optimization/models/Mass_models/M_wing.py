import math

def M_wing(b:float, c:float, sweep:float, taper_ratio:float, t_over_c:float, 
           n:float, MTOM:float, V_cruise:float) -> float:
    
    """
    b -> wing span [m] 
    c -> wing chord [m]
    sweep -> wing sweep angle [deg]
    taper_ratio -> wing taper ratio [-]
    t_over_c -> wing thickness to chord ratio [-]
    n -> ultimate load factor [-]
    MTOM -> maximum take off mass [kg]
    V_cruise -> cruise airspeed [m/s] 

    M_wing -> total mass of wing [kg]

    """
    # To imperial
    b = b * 3.2808399
    c = c * 3.2808399
    V_cruise = V_cruise * 1.94384
    rho = 1.225 * 0.0624279606 # air density at sea level [lb/ft^3]
    MTOM = MTOM * 2.20462262

    S = b *c 
    AR = b/c 
    q = 0.5 * rho * V_cruise**2

    M_WR = 0.036 * (S**0.758) * ((AR / math.cos(sweep)**2)**0.6) * (q**0.006) * (taper_ratio**0.04) * ((100 * t_over_c / (math.cos(sweep)))**-0.3) * (n*MTOM)**0.49
    M_WN = 96.948 * ((n*MTOM/10**5)**0.65) * ((AR / math.cos(sweep)**2)**0.57) * ((S/100)**0.61) * (((1 + taper_ratio)/(2*t_over_c))**0.36) * (math.sqrt(1 + V_cruise/500))**0.993

    M_wing = ((M_WR + M_WN) / 2) /  2.20462262 # Back to kg

    return M_wing
