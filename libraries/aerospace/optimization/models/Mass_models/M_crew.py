def M_crew(M_pax:float, M_lug:float, n_crew:int, LF:int=1) -> float:

    M_crew = (M_pax + M_lug) * n_crew * LF

    return M_crew

