aircraft_data = {
    "b"  : 0.96,
    "mac": 0.122,
    "m"  : 1.904,
    "S"  : 0.115,
    "Ixx": 0.01124,
    "Iyy": 0.02671,
    "Izz": 0.03144,
    "Ixz": -0.00162,

    "U0": 23,
    "theta0": 0,
    "rho": 1.226,
    "g": 9.81,

    "CL0":  0.5084158,
    "CD0":  0.02260014,
    "Cxu":  -0.040262,
    "Cxa":   0.31146,
    "Czu":  -1.61e-05,
    "CLa":   5.2576,
    "CLq":   5.0021,
    "Cmu":  -0.0091925,
    "Cma":  -0.49437,
    "Cmq":  -18.909,

    "CXde":  0.0094359,
    "CYde":  0.0,
    "CZde":  0.27839,
    "CLde":  0.0,
    "CMde":  1.0585,
    "CNde":  0.0,

    "CYb":  -0.13835,
    "CYp":  -0.028121,
    "CYr":   0.14386,
    "Clb":   0.0088073,
    "Clp":  -0.54353,
    "Clr":   0.11964,
    "Cnb":   0.065205,
    "Cnp":  -0.057495,
    "Cnr":  -0.065273,

    "CXda": -9.63e-05,
    "CYda": -0.0086239,
    "CZda":  6.56e-05,
    "CLda": -0.27709,
    "CMda": -1.45e-06,
    "CNda":  0.0031458,

    "CXdr": -1.64e-05,
    "CYdr":  0.077432,
    "CZdr":  2.78e-06,
    "CLdr": -0.0022861,
    "CMdr":  3.02e-06,
    "CNdr": -0.03853
}

boeing_data = {
    'CL0': 1.11, 
    'CD0': 0.102, 
    'CLa' : 5.7, 
    'CDa' : 0.66, 
    'Cma' : -1.26,
    'CLda' : -6.7, 
    'Cmda' : -3.2, 
    'CLq' : 5.4, 
    'Cmq' : -20.8, 
    'Czu' : -0.81 * ((165*1.68781*0.3048)/340),
    'Cxu': 0,
    'Cxa': 1.11 - 0.66,
    'CmM' : 0.27, 
    'Cmu': 0.27 * ((165*1.68781*0.3048)/340),
    'CLde' : 0.338, 
    'CMde' : -1.34, 
    'CYb' : -0.96, 
    'Clb' : -0.221, 
    'Cnb' : 0.150, 
    'Clp' : -0.45, 
    'Cnp' : -0.121,
    'Clr' : 0.101, 
    'Cnr': -0.30, 
    'CLda' : 0.0461, 
    'CNda' : 0.0064, 
    'CYdr' : 0.175,
    'CYp': 0,
    'CYr': 0,
    'CYda': 0,
    'CLdr' : 0.007, 
    'CNdr' : -0.109, 
    'S': 5500*0.3048**2, 
    'b': 195.68*0.3048, 
    'mac': 27.31*0.3048, 
    'theta0': 0,
    'm': 564000*0.453592, 
    'Iyy': 32.3e6*14.5939*0.3048**2, 
    'Izz': 45.3e6*14.5939*0.3048**2, 
    'Ixx': 14.3e6*14.5939*0.3048**2,
    'Ixz': -2.23e6*14.5939*0.3048**2, 
    'h': 0, 
    'U0':165*1.68781*0.3048, 
    'rho': 1.225, 
    'g': 9.81, 
    'theta_T': 0, 
    'dZT': 0
}

def transient_response(input_dict):

    # --- 0) Imports ---
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import control


    # --- 1) Read Aircraft Data --- 

    b = input_dict['b']
    mac = input_dict['mac']
    m = input_dict['m']
    S = input_dict['S']
    Ixx = input_dict['Ixx']
    Iyy = input_dict['Iyy']
    Izz = input_dict['Izz']
    Ixz = input_dict['Ixz']

    U0 = input_dict['U0']
    rho = input_dict['rho']
    theta0 = input_dict['theta0']
    g = input_dict['g']
    q = 0.5 * rho * U0**2 # dynamic pressure

    # Lift coefficient
    CL0 = input_dict['CL0']
    Czu  = input_dict["Czu"]
    CLa  = input_dict["CLa"]
    CLq  = input_dict["CLq"]
    CLde = input_dict["CLde"]

    # Sideforce coefficient
    CYb  = input_dict["CYb"]
    CYp  = input_dict["CYp"]
    CYr  = input_dict["CYr"]
    CYda = input_dict["CYda"]
    CYdr = input_dict["CYdr"]

    # Drag coefficient
    CD0 = input_dict['CD0']
    Cxu = input_dict['Cxu']
    Cxa  = input_dict["Cxa"]

    # Pitching moment coefficient
    Cmu  = input_dict["Cmu"]
    Cma  = input_dict["Cma"]
    Cmq  = input_dict["Cmq"]
    CMde = input_dict["CMde"]

    # Rolling moment coefficient
    Clb  = input_dict["Clb"]
    Clp  = input_dict["Clp"]
    Clr  = input_dict["Clr"]
    CLda = input_dict["CLda"]
    CLdr = input_dict["CLdr"]

    # Yawing moment coefficient
    Cnb  = input_dict["Cnb"]
    Cnp  = input_dict["Cnp"]
    Cnr  = input_dict["Cnr"]
    CNda = input_dict["CNda"]
    CNdr = input_dict["CNdr"]


    # --- 2) Convert to Dimensional Form
    # --- Longitudinal ---
    Xu = ((-q * S) / (m * U0)) * (2 * CD0 + Cxu) 
    Xw = ((q * S) / (m  * U0)) * (Cxa)
    Xq = 0
    Zu = ((-q * S) / (m * U0)) * (2 * CL0 + Czu)
    Zw = ((-q * S) / (m * U0)) * (CD0 + CLa)
    Zq = ((-q * S * mac) / (2 * m * U0)) * (CLq)
    Mu = ((q * S * mac) / (Iyy * U0)) * (Cmu)
    Mw = ((q * S * mac) / (Iyy * U0)) * (Cma)
    Mdw = 0
    Mq = ((q * S * mac**2) / (2 * Iyy * U0)) * (Cmq)
    Zde = ((-q * S) / m) * (CLde)
    Mde = ((q * S * mac) / Iyy) * (CMde)

    # Obtain remaining terms
    Mustar = Mu + Mdw * Zu
    Mwstar = Mw + Mdw * Zw
    Mqstar = Mq + Mdw * Zq
    Mthetastar = -Mdw * g * np.sin(theta0)
    Mdestar = Mde + Mdw * Zde

    # --- Lateral ---
    Yv = ((q * S) / (m * U0)) * (CYb)
    Yp = ((q * S) / (2 * m * U0)) * (CYp)
    Yr = ((q * S * b) / (2 * m * U0)) * (CYr)
    Lv = ((q * S * b) / (Ixx * U0)) * (Clb)
    Lp = ((q * S * b**2) / (2 * Ixx * U0)) * (Clp)
    Lr = ((q * S * b**2) / (2 * Ixx * U0)) * (Clr)
    Nv = ((q * S * b) / (Izz * U0)) * (Cnb)
    Np = ((q * S * b**2) / (2 * Izz * U0)) * (Cnp)
    Nr = ((q * S * b**2) / (2 * Izz * U0)) * (Cnr)
    Yda = ((q * S) / m) * (CYda)
    Ydr = ((q * S) / m) * (CYdr)
    Lda = ((q * S * b) / Ixx) * (CLda)
    Ldr = ((q * S * b) / Ixx) * (CLdr)
    Nda = ((q * S * b) / Izz) * (CNda)
    Ndr = ((q * S * b) / Izz) * (CNdr)

    # Obtain remaining terms
    Imess = Ixx * Izz / (Ixx * Izz - Ixz**2)
    I2 = Ixz / Ixx
    Lvstar = Imess * (Lv + I2 * Nv)
    Lpstar = Imess * (Lp + I2 * Np)
    Lrstar = Imess * (Lr + I2 * Nr)
    Ldrstar = Imess * (Ldr + I2 * Ndr)
    Ldastar = Imess * (Lda + I2 * Nda)
    I2 = Ixz / Izz
    Nvstar = Imess * (Nv + I2 * Lv)
    Npstar = Imess * (Np + I2 * Lp)
    Nrstar = Imess * (Nr + I2 * Lr)
    Ndrstar = Imess * (Ndr + I2 * Ldr)
    Ndastar = Imess * (Nda + I2 * Lda)

    # 3) Obtain Linearized Dimensional EoMs ---
    # --- Longitudinal ---
    # Construct system matrix
    Alon = np.matrix([[Xu, Xw, 0, -g*np.cos(theta0)],
               [Zu, Zw, U0 + Zq, -g*np.sin(theta0)],
               [Mustar, Mwstar, Mqstar, Mthetastar],
               [0, 0, 1, 0]])
    
    # Construct control matrix
    Blon = np.matrix([[0], np.radians([Zde]), np.radians([Mdestar]), [0]])
    
    # --- Lateral ---
    # Construct system matrix
    Alat = np.matrix([[Yv, Yp, -U0 + Yr, g*np.cos(theta0), 0],
               [Lvstar, Lpstar, Lrstar, 0, 0],
               [Nvstar, Npstar, Nrstar, 0, 0],
               [0, 1, np.tan(theta0), 0, 0],
               [0, 0, 1/np.cos(theta0), 0, 0]])
    
    # Construct control matrix
    Blat = np.matrix([[Ydr, Yda], [Ldrstar, Ldastar], [Ndrstar, Ndastar], [0, 0], [0, 0]])


    # --- 4) Impulse Response Analysis ---
    # --- Unit elevator input (dE = 1 deg) ---
    # Construct state space object
    LonSS = control.StateSpace(Alon, Blon, np.eye(Alon.shape[0]), np.zeros(Blon.shape))

    # Look at the first 200 seconds response to a unit impulse
    Time, [u, w, q, theta] = control.impulse_response(LonSS, T=np.linspace(0, 200, 10000))
    u, w, q, theta = u[0], w[0], q[0], theta[0]

    # Convert q and theta
    q = np.degrees(q)
    theta = np.degrees(theta)

    # Construct figure
    fig_de = make_subplots(rows=2, cols=2, subplot_titles=("Forward Speed", "Heave Velocity", "Pitch Rate", "Pitch Attitude"))

    fig_de.add_trace(go.Scatter(x = Time, y = u, showlegend=False), row=1, col=1)
    fig_de.add_trace(go.Scatter(x = Time, y = w, showlegend=False), row=1, col=2)
    fig_de.add_trace(go.Scatter(x = Time, y = q, showlegend=False), row=2, col=1)
    fig_de.add_trace(go.Scatter(x = Time, y = theta, showlegend=False), row=2, col=2)

    fig_de.update_xaxes(title_text="Time", row=1, col=1)
    fig_de.update_yaxes(title_text=f"u / (m/s)", row=1, col=1)
    fig_de.update_xaxes(title_text="Time", row=1, col=2)
    fig_de.update_yaxes(title_text=f"w / (m/s)", row=1, col=2)
    fig_de.update_xaxes(title_text="Time", row=2, col=1)
    fig_de.update_yaxes(title_text="q / (deg/s)", row=2, col=1)
    fig_de.update_xaxes(title_text="Time", row=2, col=2)
    fig_de.update_yaxes(title_text="θ / deg", row=2, col=2)


    # --- Unit rudder input (dr = 1 deg) ---
    Blat = np.radians(Blat)

    # Turn the matrices into a state space object
    LatSS = control.StateSpace(Alat, Blat, np.eye(Alat.shape[0]), np.zeros(Blat.shape))

    # Look at the first 100 seconds response to a unit impulse
    Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, 200, 10000), input=0)
    v, p, r, phi, psi = v[0], p[0], r[0], phi[0], psi[0]

    # Convert p, r, and phi
    p = np.degrees(p)
    r = np.degrees(r)
    phi = np.degrees(phi)

    # Construct figure
    fig_dr = make_subplots(rows=2, cols=2, subplot_titles=("Sideslip Velocity", "Roll Rate", "Yaw Rate", "Roll Attitude"))
    fig_dr.update_layout(title=f"Unit Rudder Input", title_x=0.5)
    fig_dr.add_trace(go.Scatter(x = Time, y = v, showlegend=False), row=1, col=1)
    fig_dr.add_trace(go.Scatter(x = Time, y = p, showlegend=False), row=1, col=2)
    fig_dr.add_trace(go.Scatter(x = Time, y = r, showlegend=False), row=2, col=1)
    fig_dr.add_trace(go.Scatter(x = Time, y = phi, showlegend=False), row=2, col=2)

    fig_dr.update_xaxes(title_text="Time", row=1, col=1)
    fig_dr.update_yaxes(title_text=f"v / (m/s)", row=1, col=1)
    fig_dr.update_xaxes(title_text="Time", row=1, col=2)
    fig_dr.update_yaxes(title_text=f"p / (deg/s)", row=1, col=2)
    fig_dr.update_xaxes(title_text="Time", row=2, col=1)
    fig_dr.update_yaxes(title_text="r / (deg/s)", row=2, col=1)
    fig_dr.update_xaxes(title_text="Time", row=2, col=2)
    fig_dr.update_yaxes(title_text="φ  / deg", row=2, col=2)


    # --- Unit aileron input (da = 1 deg) ---
    # Look at the first 100 seconds response to a unit impulse
    Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, 200, 10000), input=1)
    v, p, r, phi, psi = v[0], p[0], r[0], phi[0], psi[0]

    # Convert p, r, and phi
    p = np.degrees(p)
    r = np.degrees(r)
    phi = np.degrees(phi)

    # Construct figure
    fig_da = make_subplots(rows=2, cols=2, subplot_titles=("Sideslip Velocity", "Roll Rate", "Yaw Rate", "Roll Attitude"))
    fig_da.update_layout(title=f"Unit Aileron Input", title_x=0.5)
    fig_da.add_trace(go.Scatter(x = Time, y = v, showlegend=False), row=1, col=1)
    fig_da.add_trace(go.Scatter(x = Time, y = p, showlegend=False), row=1, col=2)
    fig_da.add_trace(go.Scatter(x = Time, y = r, showlegend=False), row=2, col=1)
    fig_da.add_trace(go.Scatter(x = Time, y = phi, showlegend=False), row=2, col=2)

    fig_da.update_xaxes(title_text="Time", row=1, col=1)
    fig_da.update_yaxes(title_text=f"v / (m/s)", row=1, col=1)
    fig_da.update_xaxes(title_text="Time", row=1, col=2)
    fig_da.update_yaxes(title_text=f"p / (deg/s)", row=1, col=2)
    fig_da.update_xaxes(title_text="Time", row=2, col=1)
    fig_da.update_yaxes(title_text="r / (deg/s)", row=2, col=1)
    fig_da.update_xaxes(title_text="Time", row=2, col=2)
    fig_da.update_yaxes(title_text="φ  / deg", row=2, col=2)

    return fig_de, fig_dr, fig_da

fig_de, fig_dr, fig_da = transient_response(aircraft_data)
fig_de.show()
fig_dr.show()
fig_da.show()







