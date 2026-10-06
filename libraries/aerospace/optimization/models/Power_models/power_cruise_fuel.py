import math 

def power_cruise_fuel(k:float, rho_alt:float, rho_sl:float, CD0:float, AR:float, e:float, n:float, b:float,
                    MTOM:float, S:float, V:float, eta_prop:float, g:float=9.81):
    
    """
    This function computes the sea-level power required for a fuel-powered turboprop aircraft for 
    a cruise mission. This sea-level power required can be used to select an engine
    
    Inputs:
    k: throttle setting for cruise from 0 to 1 [-]
    rho_alt: air density at cruise altitude [kg/m^3]
    rho_sl: air density at sea level [kg/m^3]
    CD0: parasitic drag coefficient of 3D aircraft [-]
    AR: aspect ratio of the main wing [-]
    e: Oswald's efficiency factor of the 3D wing [-]
    n: load factor during cruise [-]
    b: mass fraction at the beginning of cruise mission [-]
    MTOM: maximum take-off mass [kg]
    S: wing reference area [m^2]
    V: cruise speed [m/s]
    eta_prop: propeller efficiency [-]

    Outputs:
    P_SL: power required at sea-level [N]
    
    """

    sigma = rho_alt/rho_sl
    P_SL = (1/(k*sigma)) * (0.5*(CD0 + (1/(math.pi*AR*e))*(2*n*b*MTOM*g/(rho_alt*S*V**2))**2) * rho_alt*S*V**2)*V

    return P_SL