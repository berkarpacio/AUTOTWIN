def M_motor(P_out:float, rho_motor:float) -> float:

    """ Valid for the power range from 10 to 260 kW per motor
    
    Inputs:
    P_out -> total power required for the electric motors [W]
    rho_motor -> power density of motors [W/kg]

    Outputs:
    M_motor -> total mass of motors [kg]

    """

    M_motor = P_out / rho_motor

    return M_motor