def M_payload(M_pax:float, M_lug:float, n_seats:int, LF:int=1) -> float:

    M_payload = (M_pax + M_lug) * n_seats * LF

    return M_payload 

