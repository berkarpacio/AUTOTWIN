import numpy as np
import sympy as sp
from IPython.display import display, Math, Latex, Markdown
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import control

# --- 0) Define Aircraft Data ---
class Cherokee():
    h = 1200
    V = 50
    S = 15
    m = 1090
    Ixx = 1300 
    Izz = 1400
    Ixz = 0
    Iyy = 1700 
    CL0 = 0.543
    CD0 = 0.0615
    b = 9.11
    cbar = 1.3
    U0 = 50
    theta0=0

    Ue = V

    g = 9.80665

    Xu = -0.06728
    Zu = -0.396
    Mu = 0.0
    Xw = 0.02323
    Zw = -1.729
    Mw = -0.2772
    Xq = 0.0
    Zq = -1.6804
    Mq = -2.207
    Mdw = -0.0197
    Zde = -17.01
    Mde = -44.71

    Yv = -0.1444
    Lv = -0.1166
    Nv = 0.174
    Lp = -2.283
    Np = -1.732
    Lr = 1.053
    Nr = -1.029
    Ydr = 2.113
    Ldr = 0.6133
    Ndr = -6.583
    Lda = 3.101
    Nda = 0.0
    Yda = 0.0



def create_state_space(ac):

    """ This function extracts aircraft modes from dimensional stability derivatives. """

    # --- 1) Compute longitudinal system and control matrices --- 
    Mu_star = ac.Mu + ac.Mdw*ac.Zu
    Mw_star = ac.Mw + ac.Mdw*ac.Zw
    Mq_star = ac.Mq + ac.Mdw*ac.Ue
    Mth_star = -ac.Mdw*ac.g*np.sin(ac.theta0)
    Mde_star = ac.Mde + ac.Mdw*ac.Zde
    
    A_lon = np.array([[ac.Xu, ac.Xw, 0, -ac.g*np.cos(ac.theta0)],
                [ac.Zu, ac.Zw, ac.U0, -ac.g*np.sin(ac.theta0)],
                [Mu_star, Mw_star, Mq_star, Mth_star],
                [0, 0, 1, 0]])
    
    B_lon = np.array([[0], [ac.Zde], [Mde_star], [0]])


    # --- 2) Compute lateral system and control matrices --- 
    Imess = ac.Ixx * ac.Izz / (ac.Ixx * ac.Izz - ac.Ixz**2)
    I2 = ac.Ixz / ac.Ixx

    Lvstar = Imess * (ac.Lv + I2 * ac.Nv)
    Lpstar = Imess * (ac.Lp + I2 * ac.Np)
    Lrstar = Imess * (ac.Lr + I2 * ac.Nr)
    Ldrstar = Imess * (ac.Ldr + I2 * ac.Ndr)
    Ldastar = Imess * (ac.Lda + I2 * ac.Nda)

    I2 = ac.Ixz / ac.Izz
    Nvstar = Imess * (ac.Nv + I2 * ac.Lv)
    Npstar = Imess * (ac.Np + I2 * ac.Lp)
    Nrstar = Imess * (ac.Nr + I2 * ac.Lr)
    Ndrstar = Imess * (ac.Ndr + I2 * ac.Ldr)
    Ndastar = Imess * (ac.Nda + I2 * ac.Lda)
    
    A_lat = np.matrix([[ac.Yv, 0, -ac.U0, ac.g*np.cos(ac.theta0), 0],
                   [Lvstar, Lpstar, Lrstar, 0, 0],
                   [Nvstar, Npstar, Nrstar, 0, 0],
                   [0, 1, np.tan(ac.theta0), 0, 0],
                   [0, 0, 1/np.cos(ac.theta0), 0, 0]])
        
    B_lat = np.matrix([[ac.Ydr, ac.Yda], [Ldrstar, Ldastar], [Ndrstar, Ndastar], [0, 0], [0, 0]])


    # --- 3) Compute the eigenvalues --- 
    eigs_lon, eigv_lon = np.linalg.eig(A_lon)
    eig_lon_1 = eigs_lon[0]
    eig_lon_2 = eigs_lon[eigs_lon.real != eig_lon_1.real][0]

    eigs_lat, eigv_lat = np.linalg.eig(A_lat)
    eig_DR = eigs_lat[eigs_lat.imag > 0][0]
    other_lat_eigs = eigs_lat[eigs_lat.imag == 0]

    # --- 4) Impulse Response Analysis ---
    # --- Unit elevator input (dE = 1 deg) ---
    # Turn the matrices into a state space object
    LonSS = control.StateSpace(A_lon, np.radians(B_lon), np.eye(A_lon.shape[0]), np.zeros(B_lon.shape))

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
    # Turn the matrices into a state space object
    LatSS = control.StateSpace(A_lat, np.radians(B_lat), np.eye(A_lat.shape[0]), np.zeros(B_lat.shape))

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


    
    return eigs_lon, eigs_lat, fig_de, fig_da, fig_dr

eigs_lon, eigs_lat, fig_de, fig_da, fig_dr = create_state_space(Cherokee)
print(eigs_lon)
print(eigs_lat)
fig_da.show()
fig_de.show()
fig_dr.show()







    