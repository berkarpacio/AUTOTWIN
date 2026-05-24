def M_empty(M_wing:float, M_rotor:float, M_fuselage:float, M_motor:float, M_systems:float, M_furnish:float, M_crew:float) -> float:

    M_empty = M_wing + M_rotor + M_motor + M_fuselage + M_systems + M_furnish + M_crew

    return M_empty 

