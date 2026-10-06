import math

def M_hydrogen_turboprop_cruise(rho_sl:float, MTOM:float, S:float, V:float, AR:float, e:float, CD0:float,
            b:float, R:float, eta_prop:float, eta_elec:float, LHV:float, g:float=9.81):
    
    """
    Computes the mass of the hydrogen storage system required for a fuel cell-powered turboprop aircraft during a 
    cruise climb mission, where the cruise speed is constant while the altitude is increasing. 
    The mass returned is the total mass of the fuel storage system needed for a given cruise range.
    The powertrain consists of a fuel cell, DC/DC converter, Inverter, Electric motor, and propeller
    
    Inputs:
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
    eta_elec: power out from electric motor/power into the fuel cell [-]
    LHV: lower heating value of hydrogen in J/kg

    Outputs:
    m_fuel: mass of fuel [kg]
    
    """

    eta_tot = eta_prop * eta_elec
    CL_cruise = (2*b*MTOM*g)/(rho_sl*S*V**2)
    K = 1/(math.pi*e*AR)
    CD_cruise = CD0 + K*CL_cruise**2
    ff = math.exp(R/((LHV*eta_tot/g) * (CL_cruise/CD_cruise)))
    m_fuel = (ff*MTOM*b - MTOM*b)/ff

    return m_fuel