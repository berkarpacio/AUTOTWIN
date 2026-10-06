def M_fuselage(l_f:float, w_f:float, d_f:float, n:float, MTOM:float, V_cruise:float) -> float:

    """ 
    l_f -> fuselage length [m]
    w_f -> fuselage max width [m]
    d_f -> fuselage max depth [m]
    n -> ultimate load factor 
    MTOM -> maximum take of mass [kg]
    V_cruise -> cruise air speed at sea level [m/s]

    M_fuselage -> estimated fuselage mass [kg]

    """

    V_cruise = V_cruise * 1.94384 # convert into knots
    MTOM = MTOM * 2.20462262 # to lb
    l_f = l_f * 3.2808399 # to ft
    w_f = w_f * 3.2808399 # to ft
    d_f = d_f * 3.2808399 # to ft

    M_fuselage = (200 * ((((n*MTOM)/10**5)**0.286) * ((l_f/10)**0.857) * ((w_f + d_f)/10) * (V_cruise/100)**0.338)**1.1) / 2.20462262
    
    return M_fuselage 
