import math

def power_climb(g:float, MTOM:float, rho_climb:float, S:float, CD_climb, V_climb:float, 
                theta_climb:float, num_rotors_cruise:float, 
                eta_prop_cruise:float, eta_elec:float) -> float:
    
    """ Computes power required by ESS during climb

    Inputs:
    g -> Acceleration due to gravity [m/s^2]
    MTOM -> Maximum take-off mass of aircraft [kg]
    theta_climb -> Climb angle [rad]
    CD_climb -> Aircraft drag coefficient during climb [-]
    rho_climb -> Air density at end of climb altitude [kg/m^3]
    S -> Aircraft wing reference area [m^2]
    V_climb -> Aircraft cruise speed [m/s]
    num_rotors_cruise -> Number of rotors used in climb [-]
    eta_prop_cruise -> Propeller efficiency at climb power [-]
    eta_elec -> Total electrical efficiency at climb power [-]
        
    Outputs:
    power_elec_total -> power required by ESS during climb [W]
    
    """

    drag_climb = 0.5 * CD_climb * rho_climb * S * V_climb**2
    thrust_climb = drag_climb + MTOM * g * math.sin(theta_climb)
    power_climb = thrust_climb * V_climb
    power_rotor = power_climb / num_rotors_cruise
    power_shaft = power_rotor / eta_prop_cruise
    power_elec_total = (power_shaft / eta_elec) * num_rotors_cruise

    return power_elec_total