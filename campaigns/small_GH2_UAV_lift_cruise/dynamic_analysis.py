""" This script analyzes the dynamic behavior of an aircraft. Non-dimensional stability and control derivatives are taken
from VSP-like tools and dimensionalized using inertial, geometric, and flow properties. Measurement in VSP only
depends on the geometry, flow condition, and CG location. """

# --- Imports ---
import numpy as np
from IPython.display import display, Math, Latex, Markdown
import sympy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import control
import control.matlab
from ambiance import Atmosphere


# --- Aircraft Data ---
# Configurational data
S = 0.9655232683394387 # wing area in m^2
b = 2.9478297476759585 # wing span in m
mac = 0.32888 # mean aero chord length in m
theta_0 = 0 # trim theta in degress
m = 19.9 # aircraft mass in kg

# Estimated aircraft inertias
Ixx = 3.0   # kg*m^2, roll inertia
Iyy = 1.2   # kg*m^2, pitch inertia
Izz = 5.0   # kg*m^2, yaw inertia
Ixz = 0.0   # kg*m^2, product of inertia

# Measurement data
h = 3000 # altitude in m
U0 = 25 # free stream velocity in m/s
rho = float(np.squeeze(Atmosphere(h).density)) # air density in kg/m^3
g = 9.81  # acceleration due to gravity m/s^2
a = 328 # sonic velocity in m/s

q = 0.5 * rho * U0**2 # dynamic pressure
M = U0 / a # Mach number

# Nondimensional stability derivatives
longitudinal_derivatives = {
    # Trim / base coefficients
    "CL0": 0.649960,
    "CD0": 0.042122,
    "Cm0": -0.204048,

    # Alpha derivatives
    "CL_alpha": 4.908050,
    "CD_alpha": 0.277032,
    "Cm_alpha": -0.197031,

    # Pitch-rate derivatives
    "CL_q": 4.773761,
    "CD_q": -0.118601,
    "Cm_q": -2.717204,

    # Forward-speed derivatives
    "CL_u": 0.004872,
    "CD_u": 0.000262,
    "Cm_u": -0.001477,

    # Mach derivatives
    "CL_mach": 0.064105,
    "CD_mach": 0.003446,
    "Cm_mach": -0.019428,

    # Higher-order / unsteady derivatives
    "CL_alpha_2": 0.0,
    "CD_alpha_2": 0.0,
    "Cm_alpha_2": 0.0,

    "CL_alpha_dot": 0.0,
    "CD_alpha_dot": 0.0,
    "Cm_alpha_dot": 0.0,

    # Elevator derivatives
    "CL_de": 0.0545177,
    "CD_de": 0.0074768,
    "Cm_de": -0.1661283,
}

lateral_derivatives = {
    # Trim / base coefficients
    "CY0": -0.000001,
    "Cl0": -0.000002,
    "Cn0": 0.000001,

    # Sideslip derivatives
    "CY_beta": -0.164037,
    "Cl_beta": -0.062689,
    "Cn_beta": 0.052551,

    # Roll-rate derivatives
    "CY_p": -0.036432,
    "Cl_p": -0.582885,
    "Cn_p": -0.027864,

    # Yaw-rate derivatives
    "CY_r": 0.141493,
    "Cl_r": 0.261275,
    "Cn_r": -0.054503,

    # Alpha cross-coupling derivatives
    "CY_alpha": 0.000379,
    "Cl_alpha": 0.000112,
    "Cn_alpha": -0.000146,

    # Pitch-rate cross-coupling derivatives
    "CY_q": 0.001650,
    "Cl_q": 0.000729,
    "Cn_q": -0.000597,

    # Forward-speed derivatives
    "CY_u": 0.0,
    "Cl_u": 0.0,
    "Cn_u": 0.0,

    # Mach derivatives
    "CY_mach": 0.000001,
    "Cl_mach": 0.0,
    "Cn_mach": 0.0,

    # Higher-order / unsteady derivatives
    "CY_alpha_2": 0.0,
    "Cl_alpha_2": 0.0,
    "Cn_alpha_2": 0.0,

    "CY_alpha_dot": 0.0,
    "Cl_alpha_dot": 0.0,
    "Cn_alpha_dot": 0.0,

    # Aileron derivatives
    "CY_da": 0.0157117,
    "Cl_da": -0.1689089,
    "Cn_da": 0.0121352,

    # Rudder derivatives
    "CY_dr": -0.0576999,
    "Cl_dr": -0.0153967,
    "Cn_dr": 0.0199662,

    # Optional elevator lateral coupling
    "CY_de": 0.0000157,
    "Cl_de": -0.0000272,
    "Cn_de": -0.0000020,
}

locals().update(longitudinal_derivatives)
locals().update(lateral_derivatives)


# --- Convert to SI Units ---
SIunits = True

# --- Longitudinal Dynamics ---
# Convert to dimensional form
Xu = ((-q*S)/(m*U0)) * (2*CD0 + CD_u)
Xw = ((q*S)/(m*U0)) * (CL0 - CD_alpha)
Xq = ((-q*S*mac)/(2*m*U0)) * CD_q
Zu = ((-q*S)/(m*U0)) * (2*CL0 + CL_u)
Zw = ((-q*S)/(m*U0)) * (CD0 + CL_alpha)
Zdw = ((q*S*mac)/(2*m*U0**2)) * CL_alpha_dot
Zq = ((-q*S*mac)/(2*m*U0)) * CL_q
Mu = ((q*S*mac)/(Iyy*U0)) * Cm_u
Mw = ((q*S*mac)/(Iyy*U0)) * Cm_alpha
Mdw = ((q*S*mac**2)/(2*Iyy*U0**2)) * Cm_alpha_dot
Mq = ((q*S*mac**2)/(2*Iyy*U0)) * Cm_q
Zde = ((-q*S)/(m)) * CL_de
Mde = ((q*S*mac)/(Iyy)) * Cm_de

# Obtain remaining terms
Mustar = Mu + Mdw * Zu
Mwstar = Mw + Mdw * Zw
Mqstar = Mq + Mdw * Zq
Mthetastar = -Mdw * g * np.sin(theta_0)
Mdestar = Mde + Mdw * Zde

# Construct system matrix
Alon = np.matrix([
    [Xu, Xw, Xq, -g*np.cos(theta_0)],
    [Zu, Zw, U0 + Zq, -g*np.sin(theta_0)],
    [Mustar, Mwstar, Mqstar, Mthetastar],
    [0, 0, 1, 0]
])

print("The system matrix for longitudinal motion is ")
print(Alon)


# --- Lateral Dynamics ---
# Convert to dimensional form
Yv = ((q*S)/(m*U0)) * CY_beta
Yp = ((q*S*b)/(2*m*U0)) * CY_p
Yr = ((q*S*b)/(2*m*U0)) * CY_r
Lv = ((q*S*b)/(Ixx*U0)) * Cl_beta
Lp = ((q*S*b**2)/(2*Ixx*U0))* Cl_p
Lr = ((q*S*b**2)/(2*Ixx*U0)) * Cl_r
Nv = ((q*S*b)/(Izz*U0)) * Cn_beta
Np = ((q*S*b**2)/(2*Izz*U0)) * Cn_p
Nr = ((q*S*b**2)/(2*Izz*U0)) * Cn_r
Yda = ((q*S)/(m)) * CY_da
Ydr = ((q*S)/(m)) * CY_dr
Lda = ((q*S*b)/(Ixx)) * Cl_da
Ldr = ((q*S*b)/(Ixx)) * Cl_dr
Nda = ((q*S*b)/(Izz)) * Cn_da
Ndr = ((q*S*b)/(Izz)) * Cn_dr

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

# Construct system matrix
Alat = np.matrix([
    [Yv, Yp, Yr - U0, g*np.cos(theta_0), 0],
    [Lvstar, Lpstar, Lrstar, 0, 0],
    [Nvstar, Npstar, Nrstar, 0, 0],
    [0, 1, np.tan(theta_0), 0, 0],
    [0, 0, 1/np.cos(theta_0), 0, 0]
])

print("The system matrix for lateral-directional motion is ")
print(Alat)


# --- Eigenvalue Analysis ---
# Longitudinal
eigs_lon, eigv_lon = np.linalg.eig(Alon) # eigv_lon returns eigenvectors
print("Eigenvalues for the longitudinal system matrix")
for i in eigs_lon:
    print(i)

# Lateral
eigs_lat, eigv_lat = np.linalg.eig(Alat) # eigv_lat returns eigenvectors
print("Eignevalues for the lateral system matrix")
for i in eigs_lat:
    print(i)


# --- Transient Longitudinal Response --- 
t_analysis = 100 # [s]

# We will make a B matrix to enable us to use the control system toolbox by exciting the aircraft with 1 deg elevator input
Blon = np.matrix([[0], np.radians([Zde]), np.radians([Mdestar]), [0]])

# Turn the matrices into a state space object
LonSS = control.StateSpace(Alon, Blon, np.eye(Alon.shape[0]), np.zeros(Blon.shape))

# Look at the first 200 seconds response to a unit impulse
Time, [u, w, q, theta] = control.impulse_response(LonSS, T=np.linspace(0, t_analysis, 10000))
u, w, q, theta = u[0], w[0], q[0], theta[0]

# Convert q and theta
q = np.degrees(q)
theta = np.degrees(theta)

fig = make_subplots(rows=2, cols=2, subplot_titles=("Forward Speed", "Heave Velocity", "Pitch Rate", "Pitch Attitude"))

fig.add_trace(
    go.Scatter(x = Time, y = u, showlegend=False), row=1, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = w, showlegend=False), row=1, col=2)

fig.add_trace(
    go.Scatter(x = Time, y = q, showlegend=False), row=2, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = theta, showlegend=False), row=2, col=2)

# Make a label based upon the units
speedlabel = "m/s"
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text=f"u / ({speedlabel})", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text=f"w / ({speedlabel})", row=1, col=2)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="q / (deg/s)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_yaxes(title_text="θ / deg", row=2, col=2)
fig.show()

# Analyse AoA
alpha = np.degrees(w/(U0+u))

fig = go.Figure()
fig.add_trace(
    go.Scatter(x = Time, y = alpha, showlegend=False))

fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="$\\alpha/\\text{deg}$")
fig.show()


# --- Transient Lateral Response --- 

# We will make a B matrix to enable us to use the control system toolbox by exciting the aircraft with 1 deg rudder input
Blat = np.matrix([[Ydr, Yda], [Ldrstar, Ldastar], [Ndrstar, Ndastar], [0, 0], [0, 0]])
Blat = np.radians(Blat)

# Turn the matrices into a state space object
LatSS = control.StateSpace(Alat, Blat, np.eye(Alat.shape[0]), np.zeros(Blat.shape))

# Look at the first 200 seconds response to a unit impulse
Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, t_analysis, 10000), input=0)
v, p, r, phi, psi = v[0], p[0], r[0], phi[0], psi[0]

# Convert p, r, and phi
p = np.degrees(p)
r = np.degrees(r)
phi = np.degrees(phi)


fig = make_subplots(rows=2, cols=2, subplot_titles=("Sideslip Velocity", "Roll Rate", "Yaw Rate", "Roll Attitude"))

fig.update_layout(title=f"Unit Rudder Input", title_x=0.5)

fig.add_trace(
    go.Scatter(x = Time, y = v, showlegend=False), row=1, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = p, showlegend=False), row=1, col=2)

fig.add_trace(
    go.Scatter(x = Time, y = r, showlegend=False), row=2, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = phi, showlegend=False), row=2, col=2)

# Make a label based upon the units
speedlabel = "m/s"

fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text=f"v / ({speedlabel})", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text=f"p / (deg/s)", row=1, col=2)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="r / (deg/s)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_yaxes(title_text="φ  / deg", row=2, col=2)
fig.show()

# Let's look at aileron now 
Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, t_analysis, 10000), input=1)
v, p, r, phi, psi = v[0], p[0], r[0], phi[0], psi[0]

# Convert p, r, and phi
p = np.degrees(p)
r = np.degrees(r)
phi = np.degrees(phi)

# Visualize
fig = make_subplots(rows=2, cols=2, subplot_titles=("Sideslip Velocity", "Roll Rate", "Yaw Rate", "Roll Attitude"))
fig.update_layout(title="Unit Aileron Input", title_x=0.5)

fig.add_trace(
    go.Scatter(x = Time, y = v, showlegend=False), row=1, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = p, showlegend=False), row=1, col=2)

fig.add_trace(
    go.Scatter(x = Time, y = r, showlegend=False), row=2, col=1)

fig.add_trace(
    go.Scatter(x = Time, y = phi, showlegend=False), row=2, col=2)

# Make a label based upon the units

speedlabel = "m/s"

fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text=f"v / ({speedlabel})", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text=f"p / (deg/s)", row=1, col=2)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="r / (deg/s)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_yaxes(title_text="φ  / deg", row=2, col=2)
fig.show()