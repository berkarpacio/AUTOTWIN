def V_hydrogen(Z:float, T:float, m_H2_stored:float, P_tank:float, R:float=8.314) -> float:

    """Computes the hydrogen storage tank volume 
    
    Inputs:
    Z: hydrogen compressibility at 35 MPa and 293 K [-]
    R: universal gas constant [J/(K*mol)]
    M: molecular mass of hydrogen in [g/mol]
    T: hydrogen storage temperature in [K]
    m_H2_stored: mass of hydrogen stored [kg]

    Outputs:
    V_tank: storage tank volume [m^3]

    """
    M = 2.016
    V_tank = (Z*R*T*(m_H2_stored*1000))/(P_tank*M)

    return V_tank
