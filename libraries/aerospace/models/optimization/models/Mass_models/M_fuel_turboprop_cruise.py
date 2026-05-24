import math

def M_fuel_turboprop_cruise(SFC:float, rho_sl:float, MTOM:float, S:float, V:float, AR:float, e:float, CD0:float,
            b:float, R:float, rho_fuel_sys:float, eta_prop:float, g:float=9.81):
    
    """
    Computes the mass of the fuel system required for a fuel-powered turboprop aircraft during a cruise climb mission, 
    where the cruise speed is constant while the altitude is increasing. The mass returned is the total mass of
    the fuel system needed for a given cruise range 
    
    Inputs:
    SFC: specific fuel consumption [kg/(W*s)]
    rho_sl: air density at sea-level [kg/m^3]
    MTOM: maximum take-off mass [kg]
    S: wing reference area [m^2]
    V: cruise speed [m/s]
    AR: aspect ratio of the main wing [-]
    e: Oswald's efficiency factor of the 3D wing [-]
    CD0: parasitic drag coefficient of 3D aircraft [-]
    b: mass fraction at the beginning of cruise mission [-]
    R: cruise range [m]
    eta_prop: propeller efficiency [-]

    Outputs:
    m_fuel_sys: mass of the fuel system [kg]
    
    """

    CL_cruise = (2*b*MTOM*g)/(rho_sl*S*V**2)
    K = 1/(math.pi*e*AR)
    CD_cruise = CD0 + K*CL_cruise**2
    ff = math.exp(R/((eta_prop/SFC*g) * (CL_cruise/CD_cruise)))
    m_fuel = (ff*MTOM*b - MTOM*b)/ff
    m_fuel_sys = m_fuel / rho_fuel_sys

    return m_fuel_sys