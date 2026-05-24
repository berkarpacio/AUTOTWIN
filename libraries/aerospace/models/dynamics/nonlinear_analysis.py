"""  This script analyzses the behavior of an aircraft using nonlinear 6DOF EoMs. Aerodynamic model used in the functions
present in this script accepts nonlinear stability derivatives and aero coefficients extracted from CFD, wind tunnel, panel methods,
or other relevant means. This means that the forces and moments computed using these coefficients are not perturbational forces and moments,
they are total forces and moments. """

# --- Generic Import ---
import control
import control.matlab
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- 0) Record Aircraft Data ---
aircraft_data = {
    "b"  : 0.96,
    "mac": 0.122,
    "m"  : 1.904,
    "S"  : 0.115,
    "Ixx": 0.01124,
    "Iyy": 0.02671,
    "Izz": 0.03144,
    "Ixz": -0.00162,

    "rho": 1.226,
    "g": 9.81,

    "theta_T": 0,
    "dZT": 0,

    "CL0":  0.5084158,
    "CD0":  0.02260014,
    "Cm0": 0.005407926,
    "Cxu":  -0.040262,
    "Cxa":   0.31146,
    "Czu":  -1.61e-05,
    "CLa":   5.2576,
    "CLq":   5.0021,
    "Cmu":  -0.0091925,
    "Cma":  -0.49437,
    "Cmq":  -18.909,
    "CDa2":  0,

    "CXde":  0.0094359,
    "CYde":  0.0,
    "CZde":  0.27839,
    "CLde":  0.0,
    "Cmde":  1.0585,
    "Cnde":  0.0,

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
    "Clda": -0.27709,
    "Cmda": -1.45e-06,
    "Cnda":  0.0031458,

    "CXdr": -1.64e-05,
    "CYdr":  0.077432,
    "CZdr":  2.78e-06,
    "Cldr": -0.0022861,
    "Cmdr":  3.02e-06,
    "Cndr": -0.03853
}

hs125_data = {
    "b"  : 1,
    "mac": 2.29,
    "m"  : 7484.4,
    "S"  : 32.8,
    "Ixx": 1,
    "Iyy": 84309,
    "Izz": 1,
    "Ixz": 1,

    "rho": 1.226,
    "g": 9.81,

    "theta_T": 0,
    "dZT": -0.378,

    "CL0":  0.895,
    "CD0":  0.177,
    "Cm0": -0.046,
    "Cxu":  0,
    "Cxa":  0.895 - 0.232,
    "Czu":  0,
    "CLa":  5.01,
    "CLq":  0,
    "Cmu":  0,
    "Cma":  -1.087,
    "Cmq":  -7.055,

    "CXde":  0,
    "CYde":  0,
    "CZde":  0,
    "CLde":  0.722,
    "Cmde":  -1.88,
    "Cnde":  0.0,
    "CDa2":  1.393,

    "CYb":  0,
    "CYp":  0,
    "CYr":  0,
    "Clb":  0,
    "Clp":  0,
    "Clr":  0,
    "Cnb":  0,
    "Cnp":  0,
    "Cnr":  0,

    "CXda": 0,
    "CYda": 0,
    "CZda": 0,
    "Clda": 0,
    "Cmda": 0,
    "Cnda": 0,

    "CXdr": 0,
    "CYdr": 0,
    "CZdr": 0,
    "Cldr": 0,
    "Cmdr": 0,
    "Cndr": 0
}


# 1) --- Wrap 6DOF Nonliner EoMs ---
def nonliner_6DOF(y, t = [], T=0, input_dict={}):

    """"
    This function wraps nonlinear 6DOF equations of motion into
    a 15x1 ydot vector, which contains the derivatives of the entries in y.

    Inputs:
    y -> [U, V, W, P, Q, R, phi, theta, psi, X_E, Y_E, Z_E, dA, dE, dR] is the initial condtions of the aircraft states [SI].
    Thrust -> Thrust force applied [SI].
    input_dict -> Dictionary containing aircraft configurational data and stability derivatives [SI]

    
    Outputs: 
    ydot -> derivatives of state vector y [SI].

    """

    # Imports 
    import numpy as np
    from ambiance import Atmosphere

    # Read aircraft data
    b = input_dict['b']
    mac = input_dict['mac']
    m = input_dict['m']
    S = input_dict['S']
    Ixx = input_dict['Ixx']
    Iyy = input_dict['Iyy']
    Izz = input_dict['Izz']
    Ixz = input_dict['Ixz']

    rho = input_dict['rho']
    g = input_dict['g']

    theta_T = input_dict['theta_T']
    dZT = input_dict['dZT']

    # Lift coefficient
    CL0 = input_dict['CL0']
    CLa  = input_dict["CLa"]
    CLq  = input_dict["CLq"]
    CLde = input_dict["CLde"]

    # Side force coefficient
    CYb  = input_dict["CYb"]
    CYp  = input_dict["CYp"]
    CYr  = input_dict["CYr"]
    CYda = input_dict["CYda"]
    CYdr = input_dict["CYdr"]
    CYde = input_dict["CYde"]

    # Drag coefficient 
    CD0 = input_dict['CD0']
    Cxa  = input_dict["Cxa"]
    CXde = input_dict["CXde"]
    CXda = input_dict["CXda"]
    CXdr = input_dict["CXdr"]
    CDa2 = input_dict["CDa2"]

    # Pitching moment coefficient
    Cm0 = input_dict['Cm0']
    Cma  = input_dict["Cma"]
    Cmq  = input_dict["Cmq"]
    Cmde = input_dict["Cmde"]

    # Rolling moment coefficient
    Clb  = input_dict["Clb"]
    Clp  = input_dict["Clp"]
    Clr  = input_dict["Clr"]
    Clda = input_dict["Clda"]
    Cldr = input_dict["Cldr"]

    # Yawing moment coefficient
    Cnb  = input_dict["Cnb"]
    Cnp  = input_dict["Cnp"]
    Cnr  = input_dict["Cnr"]
    Cnda = input_dict["Cnda"]
    Cndr = input_dict["Cndr"]

    # Get initial aircraft states
    U = y[0] 
    V = y[1]
    W = y[2] 
    P = y[3]
    Q = y[4]
    R = y[5]
    phi = y[6]
    theta = y[7]
    psi = y[8]
    X_E = y[9]
    Y_E = y[10]
    Z_E = y[11]
    dA = y[12]
    dE = y[13] 
    dR = y[14]

    # Compute aerodynamic terms
    Vf = np.sqrt(U**2 + V**2 + W**2)
    alpha = np.arctan(W/U)
    beta = np.arcsin(V/Vf)
    q_bar = (Q * mac) / (2 * Vf)
    p_bar = (P * b) / (2 * Vf)
    r_bar = (R * b) / (2 * Vf)
    q_inf = 0.5 * rho * Vf**2

    # Compute aerodynamic coefficients
    CL = CL0 + CLa * alpha + CLq * q_bar + CLde * dE
    CY = CYb * beta + CYp * p_bar + CYr * r_bar + CYda * dA + CYde * dE + CYdr * dR
    CD = CD0 + (CL0 - Cxa) * alpha + CXde * dE + CXda * dA + CXdr * dR + CDa2 * alpha**2

    Cl = Clb * beta + Clp * p_bar + Clr * r_bar + Clda * dA + Cldr * dR
    Cm = Cm0 + Cma * alpha + Cmq * q_bar + Cmde * dE
    Cn = Cnb * beta + Cnp * p_bar + Cnr * r_bar + Cnda * dA + Cndr * dR

    # Compute dimensional forces and moments 
    lift = S * q_inf * CL
    Y = S * q_inf * CY
    D = S * q_inf * CD

    L = S * q_inf * b * Cl
    M = S * q_inf * mac * Cm
    N = S * q_inf * b * Cn

    # Compute derivatives
    ydot = np.zeros(15, dtype='float64')
    ydot[0] = (T*np.cos(theta_T))/m - (D*np.cos(alpha))/m + (lift*np.sin(alpha))/m -g*np.sin(theta) - Q*W + U*R
    ydot[1] = g*np.sin(phi)*np.cos(theta) + Y/m - R*U + P*W
    ydot[2] = (-lift * np.cos(alpha))/m - (D*np.sin(alpha))/m + g*np.cos(phi)*np.cos(theta) - (T*np.sin(theta_T))/m + Q*U - P*V 
    ydot[5] = (Ixx*N + Ixz*L + P*Q*(Ixx*(Ixx - Iyy) + Ixz*Izz) + Ixz*Q*R*(Iyy - Izz - Ixx)) / (Izz*(Ixx - Ixz))
    ydot[4] = ((M - T*dZT) - P*R*(Ixx - Iyy) - (P**2 - R**2)*Ixz)/Iyy
    ydot[3] = (L +  (ydot[5] + P*Q)*Izz - Q*R*(Izz - Iyy))/Ixx
    ydot[6] = P + Q*np.sin(phi)*np.tan(theta) + R*np.cos(phi)*np.tan(theta)
    ydot[7] = Q*np.cos(phi) - R*np.sin(phi)
    ydot[8] = P*np.sin(phi)/np.cos(theta) + R*np.cos(phi)/np.cos(theta)
    ydot[9] = U*np.cos(theta)*np.cos(psi) + V*np.cos(theta)*np.sin(psi) - W*np.sin(theta)
    ydot[10] = U*(-np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)) + V*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi)) + W*np.sin(phi)*np.cos(theta)
    ydot[11] = U*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi)) + V*(-np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)) + W*np.cos(phi)*np.cos(theta)
    ydot[12] = 0
    ydot[13] = 0
    ydot[14] = 0

    return ydot

# 2) --- Determine Trim State ---
def total_forces(T, dA, dE, dR, phi, theta, Vf, h, gamma, input_dict):

    """ Computes forces and moments for a given set of aircrfat states. 
    
    Inputs:
    T -> thrust [SI]
    de -> elevator deflection [deg]
    da -> aileron deflection [deg]
    dr -> rudder deflection [deg]
    phi -> roll angle [deg]
    theta -> pitch angle [deg]
    psi -> yaw angle [deg]
    Vf -> flight TAS [SI]
    h -> altitude [SI]
    gamma -> flight path angle [deg]
    input_dict -> Dictionary containing aircraft configurational data and stability derivatives [SI]
    
    Outputs:
    forces_and_moments -> 6x1 vector [X, Y, Z, L, M, N] in [SI]
    
    """

    # Imports
    import numpy as np
    from ambiance import Atmosphere

    # Read aircraft data
    b = input_dict['b']
    mac = input_dict['mac']
    m = input_dict['m']
    S = input_dict['S']
    Ixx = input_dict['Ixx']
    Iyy = input_dict['Iyy']
    Izz = input_dict['Izz']
    Ixz = input_dict['Ixz']

    rho = float(np.squeeze(Atmosphere(h/1000).density))
    g = input_dict['g']

    theta_T = input_dict['theta_T']
    dZT = input_dict['dZT']

    # Lift coefficient
    CL0 = input_dict['CL0']
    CLa  = input_dict["CLa"]
    CLq  = input_dict["CLq"]
    CLde = input_dict["CLde"]

    # Side force coefficient
    CYb  = input_dict["CYb"]
    CYp  = input_dict["CYp"]
    CYr  = input_dict["CYr"]
    CYda = input_dict["CYda"]
    CYdr = input_dict["CYdr"]
    CYde = input_dict["CYde"]

    # Drag coefficient 
    CD0 = input_dict['CD0']
    Cxa  = input_dict["Cxa"]
    CXde = input_dict["CXde"]
    CXda = input_dict["CXda"]
    CXdr = input_dict["CXdr"]
    CDa2 = input_dict["CDa2"]

    # Pitching moment coefficient
    Cm0 = input_dict['Cm0']
    Cma  = input_dict["Cma"]
    Cmq  = input_dict["Cmq"]
    Cmde = input_dict["Cmde"]

    # Rolling moment coefficient
    Clb  = input_dict["Clb"]
    Clp  = input_dict["Clp"]
    Clr  = input_dict["Clr"]
    Clda = input_dict["Clda"]
    Cldr = input_dict["Cldr"]

    # Yawing moment coefficient
    Cnb  = input_dict["Cnb"]
    Cnp  = input_dict["Cnp"]
    Cnr  = input_dict["Cnr"]
    Cnda = input_dict["Cnda"]
    Cndr = input_dict["Cndr"]

    # Compute aerodynamic terms
    alpha = theta - gamma
    beta = 0 
    q_bar = 0
    p_bar = 0
    r_bar = 0
    q_inf = 0.5 * rho * Vf**2

    # Compute aerodynamic coefficients
    CL = CL0 + CLa * alpha + CLq * q_bar + CLde * dE
    CY = CYb * beta + CYp * p_bar + CYr * r_bar + CYda * dA + CYde * dE + CYdr * dR
    CD = CD0 + (CL0 - Cxa) * alpha + CXde * dE + CXda * dA + CXdr * dR + CDa2 * alpha**2

    Cl = Clb * beta + Clp * p_bar + Clr * r_bar + Clda * dA + Cldr * dR
    Cm = Cm0 + Cma * alpha + Cmq * q_bar + Cmde * dE
    Cn = Cnb * beta + Cnp * p_bar + Cnr * r_bar + Cnda * dA + Cndr * dR

    # Compute dimensional forces and moments 
    lift = S * q_inf * CL
    Y = S * q_inf * CY
    D = S * q_inf * CD

    L = S * q_inf * b * Cl
    M = S * q_inf * mac * Cm
    N = S * q_inf * b * Cn

    # Compute total forces and moments
    F_and_M = np.zeros((6, 1))
    F_and_M[0] = T*np.cos(theta_T) + lift*np.sin(alpha) -D*np.cos(alpha) -m*g*np.sin(theta)
    F_and_M[1] = m*g*np.sin(phi)*np.cos(theta) + Y
    F_and_M[2] = -lift*np.cos(alpha) -D*np.sin(alpha) + m*g*np.cos(theta)*np.cos(phi) -T*np.sin(theta_T)
    F_and_M[3] = L
    F_and_M[4] = M - T*dZT
    F_and_M[5] = N

    return F_and_M


def trim_state(Vf, h, gamma, input_dict):

    from scipy.integrate import odeint

    ''' This function solves for the trim state of a fixed wing aircraft
    using the Newton-Raphson method. For a given speed , 
    altitude, and gamma, the function determines the trim thrust,
    theta, phi, de, da, and dr.
    
    Inputs:
    Vf -> flight TAS [SI]
    h -> altitude [SI] 
    gamma -> flight path angle [deg]
    input_dict -> Dictionary containing aircraft configurational data and stability derivatives [SI]
    
    Outputs:
    TrimState -> 6x1 vector [T, de, da, dr, theta, phi] in [N, rad, rad, rad, rad, rad]
    '''

    # Imports
    import numpy as np
    from ambiance import Atmosphere

    # First guess and increments for the jacobian (adjust accordingly)
    T = 2
    dT = 1
    da = 0*np.pi/180
    dda = .01*np.pi/180
    de = 0*np.pi/180
    dde = .01*np.pi/180
    dr = 0*np.pi/180
    ddr = .01*np.pi/180
    theta = 2*np.pi/180
    dtheta = dde
    phi = 1*np.pi/180
    dphi = dda

    # Gets the value of the functions at the initial guess
    trim = total_forces(T, da, de, dr, phi, theta, Vf, h, gamma, input_dict)

    Trimstate = np.array([[T], [da], [de], [dr], [theta], [phi]])

    itercount = 0
    while max(abs(trim)) > 1e-5:  
        itercount = itercount + 1
        # Get value of the function
        trim = total_forces(T, da, de, dr, phi, theta, Vf, h, gamma, input_dict)

        # Finite-difference Jacobian approximation (6 x 6)
        JT = np.squeeze((total_forces(T + dT, da, de, dr, phi, theta, Vf, h, gamma, input_dict) - trim) / dT)
        Jda = np.squeeze((total_forces(T, da + dda, de, dr, phi, theta, Vf, h, gamma, input_dict) - trim) / dda)
        Jde = np.squeeze((total_forces(T, da, de + dde, dr, phi, theta, Vf, h, gamma, input_dict) - trim) / dde)
        Jdr = np.squeeze((total_forces(T, da, de, dr + ddr, phi, theta, Vf, h, gamma, input_dict) - trim) / ddr)
        Jtheta = np.squeeze((total_forces(T, da, de, dr, phi, theta + dtheta, Vf, h, gamma, input_dict) - trim) / dtheta)
        Jphi = np.squeeze((total_forces(T, da, de, dr, phi + dphi, theta, Vf, h, gamma, input_dict) - trim) / dphi)
        Jac = np.transpose(np.array([JT, Jda, Jde, Jdr, Jtheta, Jphi]))

        # Get the next iteration
        Trimstate = Trimstate - np.dot(np.linalg.inv(Jac), trim)

        T = Trimstate[0]
        da = Trimstate[1]
        de = Trimstate[2]
        dr = Trimstate[3]
        theta = Trimstate[4]
        phi = Trimstate[5]

    print(f"Converged after {itercount} iterations")
    print(f'For inputs of Vf = {Vf:1.2f}m/s, h = {h/1e3:1.2f}m, gamma = {gamma:1.2f}deg\n')
    print(f'Thrust = {T[0]:1.2f}N, da = {np.degrees(da[0]):1.2f}deg, de = {np.degrees(de[0]):1.2f}deg\n')
    print(f'dr = {np.degrees(de[0]):1.2f}deg, theta = {np.degrees(theta[0]):1.2f}deg, phi = {np.degrees(phi[0]):1.2f}deg\n')

    return Trimstate


# 3) --- Simulate Aircraft ---
def simulate_nonlinear_6DOF(Vf, h, gamma, de, da, dr, input_dict):

    """ This function simulates the aircraft behavior when perturbed from a 
    trim state. """

    # Imports
    import numpy as np
    from ambiance import Atmosphere
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import math
    from scipy.integrate import odeint

    # Compute trim state
    trimstate = trim_state(Vf=Vf, gamma=gamma, h=h, input_dict=input_dict)
    T_trim = trimstate[0][0]
    da_trim = trimstate[1][0]
    de_trim = trimstate[2][0]
    dr_trim = trimstate[3][0]
    theta_trim = trimstate[4][0]
    phi_trim = trimstate[5][0]
    alpha_trim = theta_trim - gamma
    
    # Compute initial condition
    Ue = Vf*np.cos(alpha_trim)
    Ve = 0
    We = Vf*np.sin(alpha_trim)
    Pe = 0
    Qe = 0
    Re = 0
    phiE = phi_trim
    thetaE = theta_trim
    psiE = 0
    Xe = 0
    Ye = 0
    Ze = h
    daE = da_trim
    deE = de_trim
    drE = dr_trim

    # Solve numerically
    y0 = np.array([Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, Xe, Ye, Ze, daE + np.radians(da), deE + np.radians(de), drE + np.radians(dr)])
    t = np.linspace(0, 100, 1000)
    output = odeint(nonliner_6DOF, y0, t, args=(T_trim, input_dict)) 

    U = output[:, 0]
    V = output[:, 1]
    W = output[:, 2]
    P = np.degrees(output[:, 3])
    Q = np.degrees(output[:, 4])
    R = np.degrees(output[:, 5])
    phi = np.degrees(output[:, 6])
    theta = np.degrees(output[:, 7])
    psi = np.degrees(output[:, 8])
    x = output[:, 9]
    y = output[:, 10]
    z = output[:, 11]
    da = np.degrees(output[:, 12])
    de = np.degrees(output[:, 13])
    dr = np.degrees(output[:, 14])
    alpha = np.arctan2(W, U)
    FS = np.sqrt(U**2 + W**2 + V**2)

    state_vector = [U,V,W,P,Q,R,phi,theta,psi,x,y,z,da,de,dr,alpha,FS,t]

    # Plot
    fig = make_subplots(rows=5, cols=3, subplot_titles=("Forward Velocity", "Sideways Velocity", "Heave Velocity", 
                                                        "Roll Rate", "Pitch Rate", "Yaw Rate", 
                                                        "Roll Attitude", "Pitch Attitude", "Yaw Attitude",
                                                        "X_NED", "Y_NED", "Z_NED",
                                                        "da", "de", "dr"))
    fig.add_trace(go.Scatter(x = t, y = U, showlegend=False), row=1, col=1) 
    fig.add_trace(go.Scatter(x = t, y = W, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x = t, y = V, showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x = t, y = P, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x = t, y = Q, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x = t, y = R, showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x = t, y = phi, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x = t, y = theta, showlegend=False), row=3, col=2)
    fig.add_trace(go.Scatter(x = t, y = psi, showlegend=False), row=3, col=3)
    fig.add_trace(go.Scatter(x = t, y = x, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x = t, y = y, showlegend=False), row=4, col=2)
    fig.add_trace(go.Scatter(x = t, y = z, showlegend=False), row=4, col=3)
    fig.add_trace(go.Scatter(x = t, y = da, showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x = t, y = de, showlegend=False), row=5, col=2)
    fig.add_trace(go.Scatter(x = t, y = dr, showlegend=False), row=5, col=3)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text=f"U / (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text=f"V / (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=1, col=3)
    fig.update_yaxes(title_text=f"W / (m/s)", row=1, col=3)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text=f"P / (deg/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text=f"Q / (deg/s)", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=3)
    fig.update_yaxes(title_text=f"R / (deg/s)", row=2, col=3)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text=f"phi / (deg)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=2)
    fig.update_yaxes(title_text=f"theta / (deg)", row=3, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=3)
    fig.update_yaxes(title_text=f"psi / (deg)", row=3, col=3)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text=f"x / (m)", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=2)
    fig.update_yaxes(title_text=f"y / (m)", row=4, col=2)
    fig.update_xaxes(title_text="Time", row=4, col=3)
    fig.update_yaxes(title_text=f"z / (m)", row=4, col=3)
    fig.update_xaxes(title_text="Time", row=5, col=1)
    fig.update_yaxes(title_text=f"da / (deg)", row=5, col=1)
    fig.update_xaxes(title_text="Time", row=5, col=2)
    fig.update_yaxes(title_text=f"de / (deg)", row=5, col=2)
    fig.update_xaxes(title_text="Time", row=5, col=3)
    fig.update_yaxes(title_text=f"dr / (deg)", row=5, col=3)
    
    return fig, state_vector


# 4) --- Linearize EoMs Using Central Difference Method ---
def perturbational_forces(U, V, W, P, Q, R, phi, theta, psi, dA, dE, dR, h, T, input_dict):
    
    """ This function computes the total forces and moments acting on an aircraft 
    as a function of its state variables. This function will be used for numerical
    linearisation of equations of motion, using the central derivative method."""

    # Imports
    import numpy as np
    from ambiance import Atmosphere

    # Read aircraft data
    b = input_dict['b']
    mac = input_dict['mac']
    m = input_dict['m']
    S = input_dict['S']
    Ixx = input_dict['Ixx']
    Iyy = input_dict['Iyy']
    Izz = input_dict['Izz']
    Ixz = input_dict['Ixz']

    rho = float(np.squeeze(Atmosphere(h/1000).density))
    g = input_dict['g']

    theta_T = input_dict['theta_T']
    dZT = input_dict['dZT']

    # Lift coefficient
    CL0 = input_dict['CL0']
    CLa  = input_dict["CLa"]
    CLq  = input_dict["CLq"]
    CLde = input_dict["CLde"]

    # Side force coefficient
    CYb  = input_dict["CYb"]
    CYp  = input_dict["CYp"]
    CYr  = input_dict["CYr"]
    CYda = input_dict["CYda"]
    CYdr = input_dict["CYdr"]
    CYde = input_dict["CYde"]

    # Drag coefficient 
    CD0 = input_dict['CD0']
    Cxa  = input_dict["Cxa"]
    CXde = input_dict["CXde"]
    CXda = input_dict["CXda"]
    CXdr = input_dict["CXdr"]
    CDa2 = input_dict["CDa2"]

    # Pitching moment coefficient
    Cm0 = input_dict['Cm0']
    Cma  = input_dict["Cma"]
    Cmq  = input_dict["Cmq"]
    Cmde = input_dict["Cmde"]

    # Rolling moment coefficient
    Clb  = input_dict["Clb"]
    Clp  = input_dict["Clp"]
    Clr  = input_dict["Clr"]
    Clda = input_dict["Clda"]
    Cldr = input_dict["Cldr"]

    # Yawing moment coefficient
    Cnb  = input_dict["Cnb"]
    Cnp  = input_dict["Cnp"]
    Cnr  = input_dict["Cnr"]
    Cnda = input_dict["Cnda"]
    Cndr = input_dict["Cndr"]

    # Compute aerodynamic terms
    alpha = np.arctan2(W, U)
    Vf = np.sqrt(U** 2 + W**2 + V**2)
    beta = np.arcsin(V/Vf)
    q_bar = (Q * mac) / (2 * Vf)
    p_bar = (P * b) / (2 * Vf)
    r_bar = (R * b) / (2 * Vf)
    q_inf = 0.5 * rho * Vf**2

    # Compute aerodynamic coefficients
    CL = CL0 + CLa * alpha + CLq * q_bar + CLde * dE
    CY = CYb * beta + CYp * p_bar + CYr * r_bar + CYda * dA + CYde * dE + CYdr * dR
    CD = CD0 + (CL0 - Cxa) * alpha + CXde * dE + CXda * dA + CXdr * dR + CDa2 * alpha**2

    Cl = Clb * beta + Clp * p_bar + Clr * r_bar + Clda * dA + Cldr * dR
    Cm = Cm0 + Cma * alpha + Cmq * q_bar + Cmde * dE
    Cn = Cnb * beta + Cnp * p_bar + Cnr * r_bar + Cnda * dA + Cndr * dR

    # Compute dimensional forces and moments 
    lift = S * q_inf * CL
    Y = S * q_inf * CY
    D = S * q_inf * CD

    L = S * q_inf * b * Cl
    M = S * q_inf * mac * Cm
    N = S * q_inf * b * Cn

    # Compute total forces and moments
    F_and_M = np.zeros((6, 1))
    F_and_M[0] = T*np.cos(theta_T) + lift*np.sin(alpha) -D*np.cos(alpha) -m*g*np.sin(theta)
    F_and_M[1] = m*g*np.sin(phi)*np.cos(theta) + Y
    F_and_M[2] = -lift*np.cos(alpha) -D*np.sin(alpha) + m*g*np.cos(theta)*np.cos(phi) -T*np.sin(theta_T)
    F_and_M[3] = L
    F_and_M[4] = M - T*dZT
    F_and_M[5] = N

    return F_and_M


def numerical_derivatives(Vf, h,  gamma, input_dict):

    """ This approach uses a central difference small perturbation approach to estimate the numerical
    value of the stability derivatives. """

    # Imports
    import numpy as np
    from ambiance import Atmosphere

    # Compute trim state
    trimstate = trim_state(Vf=Vf, gamma=gamma, h=h, input_dict=input_dict)
    T_trim = trimstate[0][0]
    da_trim = trimstate[1][0]
    de_trim = trimstate[2][0]
    dr_trim = trimstate[3][0]
    theta_trim = trimstate[4][0]
    phi_trim = trimstate[5][0]
    alpha_trim = theta_trim - gamma
    
    # Compute initial condition
    Ue = Vf*np.cos(alpha_trim)
    Ve = 0
    We = Vf*np.sin(alpha_trim)
    Pe = 0
    Qe = 0
    Re = 0
    phiE = phi_trim
    thetaE = theta_trim
    psiE = 0
    Xe = 0
    Ye = 0
    Ze = h
    daE = da_trim
    deE = de_trim
    drE = dr_trim

    # Differences
    m = input_dict['m']
    Ixx = input_dict['Ixx']
    Ixz = input_dict['Ixz']
    Iyy = input_dict['Iyy']
    Izz = input_dict['Izz']
    g = input_dict['g']
    dUe = 1
    dWe = 1
    dVe = 1
    dPe = 0.1
    dQe = 0.1
    dRe = 0.1
    # Control deflection steps are in degrees for consistency with simulate_nonlinear_6DOF inputs
    dde_deg = 0.01
    dda_deg = 0.01
    ddr_deg = 0.01
    dde = np.radians(dde_deg)
    dda = np.radians(dda_deg)
    ddr = np.radians(ddr_deg)


    # Derivatives wrt u
    Xpu, Ypu, Zpu, Lpu, Mpu, Npu = perturbational_forces(Ue + dUe, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmu, Ymu, Zmu, Lmu, Mmu, Nmu = perturbational_forces(Ue - dUe, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xu = ((1/m) * (Xpu - Xmu)) / (2 * dUe)
    Zu = ((1/m) * (Zpu - Zmu)) / (2 * dUe)
    Mu = ((1/Iyy) * (Mpu - Mmu)) / (2 * dUe)

    # Derviatives wrt w
    Xpw, Ypw, Zpw, Lpw, Mpw, Npw = perturbational_forces(Ue, Ve, We + dWe, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmw, Ymw, Zmw, Lmw, Mmw, Nmw = perturbational_forces(Ue, Ve, We - dWe, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xw = ((1/m) * (Xpw - Xmw)) / (2 * dWe)
    Zw = ((1/m) * (Zpw - Zmw)) / (2 * dWe)
    Mw = ((1/Iyy) * (Mpw - Mmw)) / (2 * dWe)

    # Derivatives wrt v
    Xpv, Ypv, Zpv, Lpv, Mpv, Npv = perturbational_forces(Ue, Ve + dVe, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmv, Ymv, Zmv, Lmv, Mmv, Nmv = perturbational_forces(Ue, Ve - dVe, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Yv = ((1/m) * (Ypv - Ymv)) / (2 * dVe)
    Lv = ((1/Ixx) * (Lpv - Lmv)) / (2 * dVe)
    Nv = ((1/Izz) * (Npv - Nmv)) / (2 * dVe)

    # Derivatives wrt p 
    Xpp, Ypp, Zpp, Lpp, Mpp, Npp = perturbational_forces(Ue, Ve, We, Pe + dPe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmp, Ymp, Zmp, Lmp, Mmp, Nmp = perturbational_forces(Ue, Ve, We, Pe - dPe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Yp = ((1/m) * (Ypp - Ymp)) / (2 * dPe)
    Lp = ((1/Ixx) * (Lpp - Lmp)) / (2 * dPe)
    Np = ((1/Izz) * (Npp - Nmp)) / (2 * dPe)

    # Derivatives wrt q
    Xpq, Ypq, Zpq, Lpq, Mpq, Npq = perturbational_forces(Ue, Ve, We, Pe, Qe + dQe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmq, Ymq, Zmq, Lmq, Mmq, Nmq = perturbational_forces(Ue, Ve, We, Pe, Qe - dQe, Re, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xq = ((1/m) * (Xpq - Xmq)) / (2 * dQe)
    Zq = ((1/m) * (Zpq - Zmq)) / (2 * dQe)
    Mq = ((1/Iyy) * (Mpq - Mmq)) / (2 * dQe)

    # Derivatives wrt r
    Xpr, Ypr, Zpr, Lpr, Mpr, Npr = perturbational_forces(Ue, Ve, We, Pe, Qe, Re + dRe, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Xmr, Ymr, Zmr, Lmr, Mmr, Nmr = perturbational_forces(Ue, Ve, We, Pe, Qe, Re - dRe, phiE, thetaE, psiE, daE, deE, drE, h, T_trim, input_dict)
    Yr = ((1/m) * (Ypr - Ymr)) / (2 * dRe)
    Lr = ((1/Ixx) * (Lpr - Lmr)) / (2 * dRe)
    Nr = ((1/Izz) * (Npr - Nmr)) / (2 * dRe)

    # Derivatives wrt de
    Xpde, Ypde, Zpde, Lpde, Mpde, Npde = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE + dde, drE, h, T_trim, input_dict)
    Xmde, Ymde, Zmde, Lmde, Mmde, Nmde = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE - dde, drE, h, T_trim, input_dict)
    Xde = ((1/m) * (Xpde - Xmde)) / (2 * dde_deg)
    Zde = ((1/m) * (Zpde - Zmde)) / (2 * dde_deg)
    Mde = ((1/Iyy) * (Mpde - Mmde)) / (2 * dde_deg)

    # Derivatives wrt da
    Xpda, Ypda, Zpda, Lpda, Mpda, Npda = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE + dda, deE, drE, h, T_trim, input_dict)
    Xmda, Ymda, Zmda, Lmda, Mmda, Nmda = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE - dda, deE, drE, h, T_trim, input_dict)
    Yda = ((1/m) * (Ypda - Ymda)) / (2 * dda_deg)
    Lda = ((1/Ixx) * (Lpda - Lmda)) / (2 * dda_deg)
    Nda = ((1/Izz) * (Npda - Nmda)) / (2 * dda_deg)

    # Derivatives wrt dr
    Xpdr, Ypdr, Zpdr, Lpdr, Mpdr, Npdr = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE + ddr, h, T_trim, input_dict)
    Xmdr, Ymdr, Zmdr, Lmdr, Mmdr, Nmdr = perturbational_forces(Ue, Ve, We, Pe, Qe, Re, phiE, thetaE, psiE, daE, deE, drE - ddr, h, T_trim, input_dict)
    Ydr = ((1/m) * (Ypdr - Ymdr)) / (2 * ddr_deg)
    Ldr = ((1/Ixx) * (Lpdr - Lmdr)) / (2 * ddr_deg)
    Ndr = ((1/Izz) * (Npdr - Nmdr)) / (2 * ddr_deg)

    def _scalar(value):
        return float(np.squeeze(value))

    Xu = _scalar(Xu)
    Xw = _scalar(Xw)
    Zu = _scalar(Zu)
    Zw = _scalar(Zw)
    Zq = _scalar(Zq)
    Mu = _scalar(Mu)
    Mw = _scalar(Mw)
    Mq = _scalar(Mq)
    Yv = _scalar(Yv)
    Yp = _scalar(Yp)
    Yr = _scalar(Yr)
    Lv = _scalar(Lv)
    Lp = _scalar(Lp)
    Lr = _scalar(Lr)
    Nv = _scalar(Nv)
    Np = _scalar(Np)
    Nr = _scalar(Nr)
    Zde = _scalar(Zde)
    Mde = _scalar(Mde)
    Yda = _scalar(Yda)
    Ydr = _scalar(Ydr)
    Lda = _scalar(Lda)
    Ldr = _scalar(Ldr)
    Nda = _scalar(Nda)
    Ndr = _scalar(Ndr)

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

    # Construct longitudinal system matrix
    Alon = np.matrix([[Xu, Xw, 0, -g*np.cos(thetaE)],
               [Zu, Zw, Ue + Zq, -g*np.sin(thetaE)],
               [Mu, Mw, Mq, 0],
               [0, 0, 1, 0]])
    
    # Construct longitudinal control matrix
    Blon = np.matrix([[0], [Zde], [Mde], [0]])
    
    # Construct lateral system matrix
    Alat = np.matrix([[Yv, Yp, -Ue + Yr, g*np.cos(thetaE), 0],
               [Lvstar, Lpstar, Lrstar, 0, 0],
               [Nvstar, Npstar, Nrstar, 0, 0],
               [0, 1, np.tan(thetaE), 0, 0],
               [0, 0, 1/np.cos(thetaE), 0, 0]])
    
    # Construct laterla control matrix
    Blat = np.matrix([[Ydr, Yda], [Ldrstar, Ldastar], [Ndrstar, Ndastar], [0, 0], [0, 0]])

    return Alon, Blon, Alat, Blat, trimstate


# --- 5) Compare Linear vs. Nonlinear ---
de = 1
da = 0
dr = 0
Vf = 23
gamma = 0
h = 80
input_dict = aircraft_data

# Nonlinear analysis
fig, state_vector = simulate_nonlinear_6DOF(Vf=Vf, h=h, gamma=gamma, de=de, da=da, dr=dr, input_dict=input_dict)
tNonlinear = state_vector[17]
Unonlinear = state_vector[0]
Wnonlinear = state_vector[1]
Qnonlinear = state_vector[4]
Thetanonlinear = state_vector[7]

# Linear analysis
Alon, Blon, Alat, Blat, trim_state = numerical_derivatives(Vf = Vf, h = h, gamma = gamma, input_dict = input_dict)
LonSS = control.StateSpace(Alon, Blon, np.eye(Alon.shape[0]), np.zeros(Blon.shape))
LatSS = control.StateSpace(Alat, Blat, np.eye(Alat.shape[0]), np.zeros(Blat.shape))
theta_trim = trim_state[4][0]
alpha_trim = theta_trim - gamma
Ue = Vf*np.cos(alpha_trim)
We = Vf*np.sin(alpha_trim)

timeVec = np.linspace(0, 100, 1000)
Time, [u, w, q, theta] = control.forced_response(LonSS, U=de * np.ones(timeVec.shape), T=timeVec)

# Convert q and theta
q = np.degrees(q)
theta = np.degrees(theta)

# Make plots
fig = make_subplots(rows=2, cols=2, subplot_titles=("Forward Speed", "Heave Velocity", "Pitch Rate", "Pitch Attitude"))
fig.add_trace(go.Scatter(x = Time, y = u + Ue, showlegend=True, name="Linear", line_color="blue"), row=1, col=1)
fig.add_trace(go.Scatter(x = tNonlinear, y = Unonlinear, line_color="blue", showlegend=True, line={"dash":"dash"}, name="Nonlinear"), row=1, col=1)
fig.add_trace(go.Scatter(x = Time, y = w+We, showlegend=False, line_color="blue"), row=1, col=2)
fig.add_trace(go.Scatter(x = tNonlinear, y = Wnonlinear, showlegend=False, line_color="blue", line={"dash":"dash"}, name="Nonlinear"), row=1, col=2)
fig.add_trace(go.Scatter(x = Time, y = (q), showlegend=False, line_color="blue"), row=2, col=1)
fig.add_trace(go.Scatter(x = Time, y = np.degrees(Qnonlinear), showlegend=False, line_color="blue", line={"dash":"dash"}), row=2, col=1)
fig.add_trace(go.Scatter(x = Time, y = (theta), showlegend=False, line_color="blue"), row=2, col=2)
fig.add_trace(go.Scatter(x = Time, y = np.degrees(Thetanonlinear), showlegend=False, line_color="blue", line={"dash":"dash"}), row=2, col=2)

speedlabel = "m/s"
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text=f"U / ({speedlabel})", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text=f"W / ({speedlabel})", row=1, col=2)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="Q / (deg/s)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_yaxes(title_text="θ / deg", row=2, col=2)

fig.update_layout(title_text=f"Comparison of linear and nonlinear models - response to {(de):1.0f}° constant elevator input",title_x=0.5)
fig.show()
