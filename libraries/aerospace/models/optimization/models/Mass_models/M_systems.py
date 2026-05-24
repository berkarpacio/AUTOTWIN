def M_systems(n:float, MTOM:float, l_f:float, b:float) -> float: 

    """
    l_f -> fuselage length [m]
    b -> wing span [m]
    n -> ultimate load factor 
    MTOM -> maximum take of mass [kg]

    M_systems -> estimated total mass of electronics and avionics [kg]

    """
    MTOM = MTOM * 2.20462262 # to lb
    l_f = l_f * 3.2808399 # to ft
    b = b * 3.2808399 # to ft

    M_systems_R = 0.053 * (l_f**1.536) * (b**0.371) * (n*MTOM*10**-4)**0.80
    M_systems_N = 1.08 * MTOM**0.7

    M_systems = ((M_systems_N + M_systems_R) / 2) / 2.20462262

    return M_systems
