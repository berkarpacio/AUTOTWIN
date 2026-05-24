""" This script defines 3DOF and 6DOF simulations """

# --- Imports ---
import numpy as np
from ambiance import Atmosphere

# --- Define 3DOF Longitudinal Nonlinear EoMs --- 
def nonlinear_3DOF_lon_EOM(y, Thrust = 13878):

    """"
    This function wraps nonlinear 3DOF longitudinal equations of motion into
    a 7x1 ydot vector, which contains the derivatives of the entries in y.

    Inputs:
    y -> [U, W, Q, Theta, X_E, Z_E, dE] is the initial condtions of the aircraft states [SI].
    Thrust -> Thrust force applied [SI].
    
    Outputs: 
    ydot -> derivatives of state vector y [SI].

    """

    # Aircraft Parameters (all in SI)
    m=7484.4 # mass  = 7484.4kg  etc.
    s=32.8 #
    Iyy=84309
    cbar=2.29 
    dZt=-0.378
    rho=1.225 
    CD0=0.177 
    CDa=0.232 
    CDa2= 1.393
    CL0=0.895
    CLa=5.01 
    CLde=0.722
    CM0=-0.046 
    CMa=-1.087 
    CMde=-1.88 
    CMq=-7.055

    # Get initial aircraft states
    U=y[0] # Forward speed
    W=y[1] # Heave velocity
    Q=y[2] # Pitch rate
    Theta=y[3] # Pitch attitude
    X_E = y[4] # X location in NED
    Z_E = y[5] # Z location in NED
    dE = y[6] # Elevator input

    # Compute aerodynamic terms
    Vf=np.sqrt(U**2 + W**2) # Flight speed
    alpha = np.arctan(W/U) # Angle of attack
    qh = Q*cbar/Vf # Nondimensional pitch rate
    
    # Aerodynamic coefficients from Hawker model:
    CL = CL0 + CLa*alpha + CLde*dE
    CD = CD0 + CDa*alpha + CDa2*alpha*alpha
    CM = CM0 + CMa*alpha + CMde * dE + CMq*qh

    q_inf = .5*rho*Vf**2*s
    
    # Dimensional aero terms
    lift = q_inf*CL
    drag = q_inf*CD
    pm = q_inf*cbar*CM

    # Simplified nonlinear equations of motion
    ydot = np.zeros(7, dtype='float64')
    ydot[0] = -Q*W + (lift*np.sin(alpha) - drag*np.cos(alpha) + Thrust)/m - 9.80665*np.sin(Theta) # Udot
    ydot[1] = Q*U + (-lift*np.cos(alpha) - drag*np.sin(alpha))/m + 9.80665*np.cos(Theta) # Wdot
    ydot[2] = (pm - Thrust*dZt)/Iyy # M
    ydot[3] = Q # ThetaDot
    ydot[4] = U*np.cos(Theta) - W*np.sin(Theta) # X_Edot (earth axes)
    ydot[5] = U*np.sin(Theta) + W*np.cos(Theta) # Z_Edot (earth axes)
    ydot[6] = 0 # No change to elevator

    return ydot


# --- Determine Trim State --- 

# Compute total forces 
def total_forces(T, de, Theta, Vf, h, gamma):

    """ Computes forces in X and Z directions as well as the total pitching moment
    for a given set of aircrfat states. 
    
    Inputs:
    T -> thrust [SI]
    de -> elevator deflection [deg]
    Theta -> pitch angle [deg]
    Vf -> flight TAS [SI]
    h -> altitude [SI]
    gamma -> flight path angle [deg]
    
    Outputs:
    F -> 3x1 vector [X, Z, M] in [SI]
    
    """

    # Aircraft Parameters (all in SI)
    m=7484.4 
    s=32.8
    Iyy=84309
    cbar=2.29 
    dZt=-0.378
    rho=1.225 
    CD0=0.177 
    CDa=0.232 
    CDa2= 1.393
    CL0=0.895
    CLa=5.01 
    CLde=0.722
    CM0=-0.046 
    CMa=-1.087 
    CMde=-1.88 
    CMq=-7.055 
    Thrust=13878
    g = 9.80665
    qh = 0  
    rho = Atmosphere(h).density

    # Compute aerodynamic terms
    alpha = Theta - gamma
    CL = CL0 + CLa*alpha + CLde*de
    CD = CD0 + CDa*alpha + CDa2*alpha*alpha
    CM = CM0 + CMa*alpha + CMde * de + CMq*qh

    q_infs = .5*rho*Vf**2*s

    # Compute dimensional lift, drag, and pitching moment
    lift = q_infs*CL
    drag = q_infs*CD
    pm = q_infs*cbar*CM

    # Determine total forces
    F = np.zeros((3, 1))
    F[0] = T - drag*np.cos(alpha) + lift*np.sin(alpha) - m*g*np.sin(Theta)
    F[1] = -lift*np.cos(alpha) -drag*np.sin(alpha) + m*g*np.cos(Theta)
    F[2] = pm - T*dZt

    return F


# Solve for the trim state using Newton-Raphson method
def TrimState(Vf = 120 * 0.5144444, h = 0, gamma = 0):
    ''' This function solves for the longitudinal trim of 
    the HS125 using a Newton-Raphson method. For a given speed, 
    altitude, and gamma, the function determines the trim thrust,
    theta, and de. 
    
    Inputs:
    Vf -> flight speed [SI]
    h -> altitude [SI] 
    gamma -> flight path angle [deg]
    
    Outputs:
    TrimState -> 3x1 vector [T, de, Theta] in [N, rad, rad]
    '''

    # First guess and increments for the jacobian <---- may need to adjust these
    T = 15000
    dT = 1
    de = 0*np.pi/180
    dde = .01*np.pi/180
    Theta = 2*np.pi/180
    dTheta = dde

    # Gets the value of the functions at the initial guess
    trim = total_forces(T, de, Theta, Vf, h, gamma)

    Trimstate = np.array([[T], [de], [Theta]])

    itercount = 0
    while max(abs(trim)) > 1e-5:  
        itercount = itercount + 1
        # Get value of the function
        trim = total_forces(T, de, Theta, Vf, h, gamma)


        # Get the Jacobian approximation (3 x 3)
        JT = np.squeeze(total_forces(T + dT, de, Theta, Vf, h, gamma)/dT)
        Jde = np.squeeze(total_forces(T, de + dde, Theta, Vf, h, gamma)/dde)
        JTheta = np.squeeze(total_forces(T, de, Theta + dTheta, Vf, h, gamma)/dTheta)
        Jac = np.transpose(np.array([JT, Jde, JTheta]))

        # Get the next iteration
        Trimstate = Trimstate - np.dot(np.linalg.inv(Jac), trim)

        T = Trimstate[0]
        de = Trimstate[1]
        Theta = Trimstate[2]
    
    print(f"Converged after {itercount} iterations")
    print(f'For inputs of Vf = {Vf:1.2f}m/s, h = {h/1e3:1.2f}km, gamma = {gamma:1.2f}deg\n')
    print(f'Thrust = {T[0]/1e3:1.2f}kN, de = {np.degrees(de[0]):1.2f}deg, Theta = {np.degrees(Theta[0]):1.2f}deg\n')

    return Trimstate

Trim = TrimState()
print(Trim)