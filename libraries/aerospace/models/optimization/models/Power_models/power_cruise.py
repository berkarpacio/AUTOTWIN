import math

def power_cruise(MTOM:float, b_cruise:float, AR:float, e:float, CD0:float, rho_cruise:float, S:float,
                 V_cruise:float, R_cruise:float, num_rotors_cruise:float, eta_cruise:float) -> float:
    
    """ Computes power required by ESS for cruise 

    Inputs:
    
    
    Outputs:
    power_cruise -> power required during cruise [W]

    """
    g = 9.81
    K = 1 / (math.pi * AR * e)
    power_cruise = (0.5 * (CD0 + K * ((2 * b_cruise * MTOM * g) / (rho_cruise * S * V_cruise**2))**2) * rho_cruise * S * V_cruise**2) * V_cruise
    thrust_cruise = power_cruise / V_cruise
    thrust_prop = thrust_cruise / num_rotors_cruise
    A_prop = math.pi * R_cruise**2
    sigma = thrust_prop / A_prop
    v_i = math.sqrt(sigma / (2 * rho_cruise))
    power_induced = ((thrust_prop * v_i) / eta_cruise) * num_rotors_cruise
    power_total = power_cruise + power_induced

    return power_total