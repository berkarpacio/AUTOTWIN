import math

def M_fuel_jet_cruise(TSFC:float, rho_cruise:float, MTOM:float, S:float, V:float, AR:float, e:float, CD0:float,
            b:float, R:float, g:float=9.81):
    
    """
    Computes the mass of the fuel required for a jet powered aircraft during a cruise climb mission, 
    where the cruise speed is constant while the altitude is increasing. The mass returned is the total mass of
    the fuel needed for a given cruise range 
    
    Inputs:
    TSFC: thrust-specific fuel consumption [kg/(N*s)]
    rho_cruise: air density at cruise altitude [kg/m^3]
    MTOM: maximum take-off mass [kg]
    S: wing reference area [m^2]
    V: cruise speed [m/s]
    AR: aspect ratio of the main wing [-]
    e: Oswald's efficiency factor of the 3D wing [-]
    CD0: parasitic drag coefficient of 3D aircraft [-]
    b: mass fraction at the beginning of cruise mission [-]
    R: cruise range [m]

    Outputs:
    m_fuel_sys: mass of the fuel system [kg]
    
    """


    CL_cruise = (2*b*MTOM*g)/(rho_cruise*S*V**2)
    K = 1/(math.pi*e*AR)
    CD_cruise = CD0 + K * CL_cruise**2
    ff = math.exp(R/((V/(TSFC*g)) * (CL_cruise/CD_cruise)))
    m_fuel = (ff*MTOM*b - MTOM*b)/ff

    return m_fuel