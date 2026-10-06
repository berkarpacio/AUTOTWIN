def M_battery(E_trip:float, E_reserve:float, rho_bat:float, e_usable:float) -> float:

    """ 
    E_trip -> total energy required for mission [Wh]
    E_reserve -> total energy required for reserves [Wh]
    rho_bat -> battery ESS energy density [Wh/kg]
    e_usable -> percentage of usable energy [-] 
    M_bat -> battery ESS mass [kg]
    
    """
    M_bat = (E_trip + E_reserve) / (rho_bat * e_usable) 

    return M_bat 