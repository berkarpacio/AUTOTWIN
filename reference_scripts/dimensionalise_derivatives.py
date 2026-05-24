""" This script analyzes the dynamic behavior of Boeing 747 using data extracted 
from a report that presents coefficients related to a B-747 in power approach configuration. """

# --- Imports ---
import numpy as np
from IPython.display import display, Math, Latex, Markdown
import sympy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import control
import control.matlab


# --- Aircraft Data ---

# Configurational data
S = 5500 # wing area in ft^2
b = 195.68 # wing span in ft
c = 27.31 # mean aero chord length in ft
theta_0 = 0 # trim theta in degress
m = 564000 # aircraft mass in lb
Iyy = 32.3e6 # slug-ft^2
Ixx = 14.3e6 # slug-ft^2 
Izz = 45.3e6 # slug-ft^2
Ixz = -2.23e6 # slug-ft^2

# Measurement data
h = 0 # altitude in ft
U0 = 165 * 1.68781 # free stream velocity in ft/s
rho = 1.225 # air density in kg/m^3
g = 9.80665  # acceleration due to gravity m/s^2
a = 340 # sonic velocity in m/s


# Nondimensional stability derivatives
B747_lon_ders = {'C_L': 1.11, 'C_D': 0.102, 'C_L_a' : 5.7, 'C_D_a' : 0.66, 'C_m_a' : -1.26,
                 'C_L_da' : -6.7, 'C_m_da' : -3.2, 'C_L_hq' : 5.4, 'C_m_hq' : -20.8, 'C_L_M' : -0.81,
                'C_m_M' : 0.27, 'C_L_de' : 0.338, 'C_m_de' : -1.34}

B747_lat_ders = {'C_y_b' : -0.96, 'C_l_b' : -0.221, 'C_n_b' : 0.150, 'C_l_hp' : -0.45, 'C_n_hp' : -0.121,
                'C_l_hr' : 0.101, 'C_n_hr': -0.30, 'C_l_da' : 0.0461, 'C_n_da' : 0.0064, 'C_y_dr' : 0.175,
                'C_l_dr' : 0.007, 'C_n_dr' : -0.109}

locals().update(B747_lon_ders)
locals().update(B747_lat_ders)


# --- Convert to SI Units ---
SIunits = True

# Conversion factors
ft_to_metre = 0.3048
lb_to_kg = 0.453592
slug_to_kg = 14.5939
MOIconvert = slug_to_kg*ft_to_metre**2

# Convert
U0 = U0 * ft_to_metre
S = S * ft_to_metre**2
b = b * ft_to_metre
c = c * ft_to_metre
m = m * lb_to_kg
Ixx = Ixx * MOIconvert
Iyy = Iyy * MOIconvert
Izz = Izz * MOIconvert
Ixz = Ixz * MOIconvert 
q = 0.5 * rho * U0**2 # dynamic pressure
M = U0 / a # Mach number


# --- Longitudinal Dynamics ---

# Convert to dimensional form
Xu = -q * S / m / U0 * (2 * C_D) # No C_D_M term so assumed zero
Xw = q * S / m / U0 * (C_L - C_D_a) # 
Xq = 0 # No CDq term given in the table so assumed zero
Zu = -q * S / m / U0 * (2 * C_L + M * C_L_M)
Zw = -q * S / m / U0 * (C_D + C_L_a)
Zdw = q * S * c / m / 2 / U0**2 * C_L_da # This is a NEW term for us, but since it was given as C_L_da, must be included
Zq = -q * S * c / 2 / m / U0 * C_L_hq
Mu = q * S * c / Iyy / U0 * M * C_m_M
Mw = q * S * c / Iyy / U0 * C_m_a
Mdw = q * S * c**2 / 2 / Iyy / U0**2 * C_m_da
Mq = q * S * c**2 / 2 / Iyy / U0 * C_m_hq
Zde = -q * S / m * C_L_de
Mde = q * S * c / Iyy * C_m_de

# Obtain remaining terms
Mustar = Mu + Mdw * Zu
Mwstar = Mw + Mdw * Zw
Mqstar = Mq + Mdw * Zq
Mthetastar = -Mdw * g * np.sin(theta_0)
Mdestar = Mde + Mdw * Zde

# Construct system matrix
Alon = np.matrix([[Xu, Xw, 0, -g*np.cos(theta_0)],
               [Zu, Zw, U0 + Zq, -g*np.sin(theta_0)],
               [Mustar, Mwstar, Mqstar, Mthetastar],
               [0, 0, 1, 0]])

print("The system matrix for longitudinal motion is ")
print(Alon)


# --- Lateral Dynamics ---

# Convert to dimensional form
Yv = q * S / m / U0 * C_y_b
Yp = 0
Yr = 0
Lv = q * S * b / Ixx / U0 * C_l_b
Lp = q * S * b**2 / 2/ Ixx / U0 * C_l_hp
Lr = q * S * b**2 / 2/ Ixx / U0 * C_l_hr
Nv = q * S * b / Izz / U0 * C_n_b
Np = q * S * b**2 / 2 / Izz / U0 * C_n_hp
Nr = q * S * b**2 / 2 / Izz / U0 * C_n_hr
Yda = 0
Ydr = q * S / m * C_y_dr
Lda = q * S * b / Ixx * C_l_da
Ldr = q * S * b / Ixx * C_l_dr
Nda = q * S * b / Izz * C_n_da
Ndr = q * S * b / Izz * C_n_dr

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
Alat = np.matrix([[Yv, 0, -U0, g*np.cos(theta_0), 0],
               [Lvstar, Lpstar, Lrstar, 0, 0],
               [Nvstar, Npstar, Nrstar, 0, 0],
               [0, 1, np.tan(theta_0), 0, 0],
               [0, 0, 1/np.cos(theta_0), 0, 0]])

print("The system matrix for lateral-directional motion is ")
print(Alat)


# --- Transient Longitudinal Response --- 

# We will make a B matrix to enable us to use the control system toolbox by exciting the aircraft with 1 deg elevator input
Blon = np.matrix([[0], np.radians([Zde]), np.radians([Mdestar]), [0]])

# Turn the matrices into a state space object
LonSS = control.StateSpace(Alon, Blon, np.eye(Alon.shape[0]), np.zeros(Blon.shape))

# Look at the first 200 seconds response to a unit impulse
Time, [u, w, q, theta] = control.impulse_response(LonSS, T=np.linspace(0, 200, 10000))
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
Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, 200, 10000), input=0)
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
if SIunits:
    speedlabel = "m/s"
else:
    speedlabel = "ft/s"

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
Time, [v, p, r, phi, psi] = control.impulse_response(LatSS, T=np.linspace(0, 200, 10000), input=1)
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
if SIunits:
    speedlabel = "m/s"
else:
    speedlabel = "ft/s"

fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text=f"v / ({speedlabel})", row=1, col=1)
fig.update_xaxes(title_text="Time", row=1, col=2)
fig.update_yaxes(title_text=f"p / (deg/s)", row=1, col=2)
fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="r / (deg/s)", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=2)
fig.update_yaxes(title_text="φ  / deg", row=2, col=2)
fig.show()