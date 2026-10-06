def M_rotor(num_hover:int, num_cruise:int, R_rotor_cruise:float, R_rotor_hover:float, k_rotor:float) -> float:
    
    """
    num_hover -> number of rotors for hover [-]
    num_cruise -> number of rotors for cruise [-]
    R_rotor_cruise -> Cruise rotor radius [m]
    R_rotor_hover -> Hover rotro radius [m]
    k_rotor -> calibration factor [-] 

    M_rotor -> total mass of all rotors [kg]
    
    """
    M_rotor = (k_rotor * ((num_cruise * (0.7484 * R_rotor_cruise**1.2 - 0.0403 * R_rotor_cruise)) + 
        (num_hover * (0.7484 * R_rotor_hover**1.2 - 0.0403 * R_rotor_hover))))

    return M_rotor