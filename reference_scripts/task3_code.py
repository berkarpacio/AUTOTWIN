### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt

### --- Data ---
CL_at_CD_min = 0.15 # Lift coefficient of aircraft at minimum drag coefficient
CD_min = 0.022 # Parasitic drag coefficient
CL_0 = 0.4 # Lift coefficient of aircraft at zero alpha
CL_alpha = 5.7 # Lift coefficient vs. alpha curve slope [1/rad]
alpha_stall = 15 # Angle of attack at stall [deg]
CL_max = CL_0 + CL_alpha * math.radians(alpha_stall) # Maximum lift coefficient
K = 0.035 # Induced drag coefficient
MTOM = 2500 # Aircraft maximum take-off mass [kg]
S_ref = 20 # Wing reference area [m^2]
g = 9.81 # Acceleration due to gravity [m/s^2]
W = MTOM * g # Aircraft maximum take-off weight [N]
P_cruise = 280 * 745.7 # Cruise power [W]
throttle_cruise = 0.8 # Cruise throttle setting
eta_prop = 0.8 # Propeller efficiency

# Initialize the altitude vector in meters
alt_vector = np.arange(0, 8000, 10)

# Initialize the speed vector for ceiling calculation
v_vect = np.arange(1, 90, 1) 

# Loop through the altitude vector to find maximum rate of climb at each altitude
R_C_max = []
for alt in alt_vector:
    rho_sl = float(np.squeeze(Atmosphere(0).density))
    rho = float(np.squeeze(Atmosphere(alt).density))
    sigma = rho/rho_sl
    P_A = P_cruise * throttle_cruise * eta_prop * sigma
    P_A_vect = np.full(v_vect.shape, P_A)
    P_R_vect = 0.5 * (CD_min + K*(2*W/(rho*S_ref*v_vect**2) - CL_at_CD_min)**2)*rho*S_ref*v_vect**3
    P_excess = P_A_vect - P_R_vect
    finite_mask = np.isfinite(P_excess)
    idx_max = np.argmax(np.where(finite_mask, P_excess, -np.inf))
    V_excess_max = v_vect[idx_max]
    P_excess_max = P_excess[idx_max]
    R_C_max.append(P_excess_max/W)

R_C_max = np.array(R_C_max)

# Find the service ceiling where max rate of climb drops to zero
ceiling_alt = alt_vector[-1]
if np.any(R_C_max <= 0) and np.any(R_C_max > 0):
    idx_zero = np.where(R_C_max <= 0)[0][0]
    if idx_zero == 0:
        ceiling_alt = alt_vector[0]
    else:
        ceiling_alt = np.interp(0, R_C_max[idx_zero-1:idx_zero+1], alt_vector[idx_zero-1:idx_zero+1])

# Rebuild altitude vector up to ceiling for flight envelope
alt_vector_ceiling = np.arange(0, ceiling_alt + 1, 10)

# Minimum and maximum flight speed vectors
V1 = np.full(alt_vector_ceiling.shape, np.nan)
V2 = np.full(alt_vector_ceiling.shape, np.nan)

# Loop through the altitude vector to solve for the minimum and maximum possible flight speeds 
for i, altitude in enumerate(alt_vector_ceiling):

    rho_sl = float(np.squeeze(Atmosphere(0).density))
    rho = float(np.squeeze(Atmosphere(altitude).density))
    sigma = rho/rho_sl
    P_available = P_cruise * throttle_cruise * sigma * eta_prop
    V_stall = math.sqrt(2*W/(rho*S_ref*CL_max))

    A = 2 * W / (rho * S_ref) 
    B = CL_at_CD_min
    C = 0.5 * S_ref * rho

    c4 = C * (CD_min + K * B**2)
    c3 = 0.0
    c2 = -rho * S_ref * K * A * B
    c1 = -P_available
    c0 = C * K * A**2

    coeffs = [c4, c3, c2, c1, c0]

    # Solve the quartic equation
    roots_v = np.roots(coeffs)
    real_positive_roots = np.sort([
        root.real
        for root in roots_v
        if np.isclose(root.imag, 0.0, atol=1e-1) and root.real > 0
    ])

    # Sort the real roots
    if len(real_positive_roots) > 0:
        if real_positive_roots[0] > V_stall:
            V1[i] = real_positive_roots[0]
        else:
            V1[i] = V_stall
    if len(real_positive_roots) > 1:
        V2[i] = real_positive_roots[1]

# Sort the speeds into an array for plotting
Vs = np.concatenate((V1, np.flip(V2)))
alt_envelope = np.concatenate((alt_vector_ceiling, np.flip(alt_vector_ceiling)))

# Create the service ceiling figure
plt.figure()
plt.plot(alt_vector, R_C_max, label="Max rate of climb")
plt.axvline(ceiling_alt, color="tab:red", linestyle="--", label=f"Ceiling ≈ {ceiling_alt:.0f} m")
plt.title("Max Rate of Climb vs Altitude", fontsize=18)
plt.xlabel("Altitude (m)", fontsize=14)
plt.ylabel("Rate of climb (m/s)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

# Create the flight envelope figure
plt.figure()
plt.plot(Vs, alt_envelope)
plt.title("Flight Envelope", fontsize=18)
plt.xlabel("TAS (m/s)", fontsize=14)
plt.ylabel("Altitude (m)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

