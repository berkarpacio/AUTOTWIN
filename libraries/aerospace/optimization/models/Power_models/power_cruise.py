def power_cruise(CD_cruise:float, rho_cruise:float, S:float,
                 V_cruise:float, num_rotors_cruise:float, eta_prop_cruise:float, eta_elec:float) -> float:
    
    """ Computes power required by ESS for cruise 

    Inputs:
    CD_cruise -> Aircraft drag coefficient during cruise [-]
    rho_cruise -> Air density at cruise altitude [kg/m^3]
    S -> Aircraft wing reference area [m^2]
    V_cruise -> Aircraft cruise speed [m/s]
    num_rotors_cruise -> Number of rotors used in cruise [-]
    eta_prop_cruise -> Propeller efficiency at cruise power [-]
    eta_elec -> Total electrical efficiency at cruise power [-]
    
    
    Outputs:
    power_elec_total -> power required by ESS during cruise [W]

    """

    drag_cruise = 0.5 * CD_cruise *  rho_cruise * S * V_cruise**2
    power_cruise = drag_cruise * V_cruise
    power_rotor = power_cruise / num_rotors_cruise
    power_shaft = power_rotor / eta_prop_cruise
    power_elec_total = (power_shaft / eta_elec) * num_rotors_cruise

    return power_elec_total