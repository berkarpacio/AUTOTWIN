def M_motor_cruise(P_climb:float, eta_p_cruise:float) -> float:

    """ Valid for the power range from 10 to 260 kW per motor

    P_climb -> total power required for climb [W]
    eta_p_cruise -> total propulsive efficiency during cruise [-]

    M_motor -> total mass of cruise motors [kg]

    """
    # Check this model, it gives low power density
    M_motor_cruise = 0.6756 * (P_climb/(eta_p_cruise * 745.7)) ** 0.783

    # M_motor = 3.1 * (10**-4) * P_climb
    return M_motor_cruise