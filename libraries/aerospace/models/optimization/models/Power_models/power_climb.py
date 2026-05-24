import math

def power_climb(MTOM:float, b_climb:float, AR:float, e:float, CD0:float, rho_climb:float, S:float,
                 V_climb:float, theta_climb:float, eta_cruise:float, R_cruise:float, num_rotors_cruise:float) -> float:
    
    """ Computes power required by ESS during climb

    Inputs:
    
    
    Outputs:
    power_total -> power required during climb [W]

    """
    g = 9.81
    K = 1 / (math.pi * AR * e)
    power_climb = (0.5 * (CD0 + K * ((2 * b_climb * MTOM * g * math.cos(theta_climb)) / (rho_climb * S * V_climb**2))**2) * rho_climb * S * V_climb**2 + b_climb * MTOM * g * math.sin(theta_climb)) * V_climb
    thrust_climb = power_climb / V_climb
    thrust_prop = thrust_climb / num_rotors_cruise
    A_prop = math.pi * R_cruise**2
    sigma = thrust_prop / A_prop
    v_i = math.sqrt(sigma / (2 * rho_climb))
    power_induced = ((thrust_prop * v_i) / eta_cruise) * num_rotors_cruise
    power_total = power_climb + power_induced

    return power_total