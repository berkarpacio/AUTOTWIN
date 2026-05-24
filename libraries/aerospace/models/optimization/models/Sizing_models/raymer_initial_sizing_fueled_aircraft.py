def size_fueled_jet_aircraft(m_crew, m_payload, a, C, K_vs, V_cruise, TSFC_cruise, TSFC_loiter, AR, R, E, K_ld, Swet_over_Sref,
                              TO_weight_fraction = 0.97, climb_weight_fraction = 0.985,
                             landing_weight_fraction = 0.995, reserve_fuel = 0.06, g = 9.81):
    
    # Imports
    import math
    
    # L/D_max estimation
    L_over_D_max_cruise = 0.866 * K_ld * math.sqrt(AR / Swet_over_Sref)
    L_over_D_max_loiter = K_ld * math.sqrt(AR / Swet_over_Sref)

    # Cruise mission segment weight fraction
    cruise_fraction = math.exp((-R * TSFC_cruise * g)/(V_cruise * L_over_D_max_cruise))

    # Loiter mission segment weight fraction
    loiter_fraction = math.exp((-E * TSFC_loiter * g)/(L_over_D_max_loiter))

    # Fuel fraction computation
    mission_fraction = cruise_fraction * loiter_fraction * TO_weight_fraction * climb_weight_fraction * landing_weight_fraction
    mf_over_mo = (1 + reserve_fuel) * (1 - mission_fraction)

    # System of equations
    # me_over_mo = a * K_vs * m_o**C
    # m_o = (m_crew + m_payload)/(1 - mf_over_mo - me_over_mo)
    # Rearranged root form:
    # f(m_o) = m_o * (1 - mf_over_mo - a*K_vs*m_o**C) - (m_crew + m_payload) = 0
    m_fixed = m_crew + m_payload

    def residual(m_o):
        return m_o * (1.0 - mf_over_mo - (a * K_vs * (m_o ** C))) - m_fixed

    # Physically valid solution requires m_o > m_fixed and positive structural denominator.
    if m_fixed <= 0:
        raise ValueError("m_crew + m_payload must be positive.")
    if mf_over_mo >= 1:
        raise ValueError("Invalid fuel fraction: mf_over_mo must be < 1.")

    # Bracket search for a positive root.
    lo = max(m_fixed * 1.0001, 1e-6)
    hi = max(lo * 2.0, 10.0)
    f_lo = residual(lo)
    f_hi = residual(hi)
    for _ in range(80):
        if f_lo == 0.0:
            m_o = lo
            break
        if f_hi == 0.0:
            m_o = hi
            break
        if f_lo * f_hi < 0.0:
            m_o = None
            break
        hi *= 2.0
        f_hi = residual(hi)
    else:
        raise RuntimeError("Could not bracket a valid m_o root with provided inputs.")

    # Bisection solve on the bracket.
    if 'm_o' not in locals() or m_o is None:
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if abs(f_mid) < 1e-10 or abs(hi - lo) / max(mid, 1.0) < 1e-10:
                m_o = mid
                break
            if f_lo * f_mid < 0.0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        else:
            m_o = 0.5 * (lo + hi)

    return m_o

m_crew = 363 # kg
m_payload = 4536 # kg
loiter_time = 3 * 3600 + (1/3) * 3600 # s
cruise_range = 2778 * 2 * 10**3 # m
V_cruise = 596.9 * 0.3048 # m/s
TSFC_cruise = 14.1 * 1/(10**6) # kg/(N*s)
TSFC_loiter = 11.3 * 1/(10**6) # kg/(N*s)
a = 0.93
C = -0.07
K_vs = 1
AR = 7
K_ld = 14.19
Swet_over_Sref = 5.5

m_o = size_fueled_jet_aircraft(m_crew, m_payload, a, C, K_vs, V_cruise, TSFC_cruise, TSFC_loiter, AR, R=cruise_range, E=loiter_time, K_ld=K_ld, Swet_over_Sref=Swet_over_Sref)
print(m_o*2.20462262)