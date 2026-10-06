import math
def power_hover(MTOM:float, rho_hover:float, R_prop_hover:float, num_rotors_hover:float, eta_prop_hover:float, eta_elec:float) -> float:
    
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
    T_rotor = (MTOM * g) / num_rotors_hover
    A_rotor = math.pi * R_prop_hover**2
    disk_loading = T_rotor / A_rotor
    power_rotor = T_rotor * math.sqrt(disk_loading/(2 * rho_hover))
    power_shaft = power_rotor / eta_prop_hover
    power_elec_total = (power_shaft / eta_elec) * num_rotors_hover

    return power_elec_total
