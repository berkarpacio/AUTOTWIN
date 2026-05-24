def M_engine(T_to_W:float, T_cruise:float) -> float: 

    """ 
    T_to_W -> thrust to weight ratio of the cruise turbofan engine [-] 
    T_cruise -> thrust required for cruise [N] 
    
    M_engine -> mass of cruise engine [kg] 
    
    """
    M_engine = (T_cruise/T_to_W) / 9.81 

    return M_engine 

