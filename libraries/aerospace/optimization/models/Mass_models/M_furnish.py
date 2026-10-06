def M_furnish(MTOM:float, n_crew:int, V_cruise:float, b:float, c:float) -> float:

    """
    b -> Wing span [m]
    c -> Wing chord length [m]
    MTOM -> Max take of mass [kg]
    n_crew -> Number of crew members [-]
    V_cruise -> Cruise airspeed [m/s]

    M_furnish -> Mass of furnishing [kg]

    """
    

    # To imperial
    b = b * 3.2808399
    c = c * 3.2808399
    V_cruise = V_cruise * 1.94384
    rho = 1.225 * 0.0624279606 # air density at sea level [lb/ft^3]
    MTOM = MTOM * 2.20462262

    S = b *c 
    q = 0.5 * rho * V_cruise**2

    M_furn_r = 0.0582 * MTOM - 65
    M_furn_n = 34.5 * n_crew * q ** 0.25

    M_furnish = ((M_furn_r + M_furn_n) / 2) / 2.20462262

    return M_furnish