def MTOM_calc(M_empty:float, M_payload:float, M_battery:float, M_engine:float, M_hydrogen:float) -> float:

    MTOM = M_empty + M_payload + M_battery + M_engine + M_hydrogen

    return MTOM 

