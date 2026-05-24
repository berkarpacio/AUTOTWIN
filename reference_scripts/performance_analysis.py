### This script analyzes the performance of fixed wing aircraft with a power engine ### 

### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt

### --- Aircraft Parameters (Cessna 172S)--- ###
MTOM = 1157 # [kg]
g = 9.81
W = MTOM * g
m_fuel = 144 # [kg]
m_final = MTOM - m_fuel
SFC = 0.28 / (3600 * 1000) # [kg/W/s]
eta_prop = 0.7 
P_SL = 134 # sea-level power available [kW]
S = 16.17 # [m^2]
CD = np.array([0.033099, 0.035185, 0.035214, 0.041961, 0.051143,\
0.064192, 0.080106, 0.096683, 0.104613, 0.11364, 0.121425, 0.130259, 0.138232])
CL = np.array([0.1454, -0.09219, 0.38303, 0.6238, 0.86305, 1.09772,\
1.31082, 1.45757, 1.51784, 1.55342, 1.58437, 1.60452, 1.58455])
K = 0.035
CD0 = 0.033
CL0 = 0.14
CLmax = max(CL)
alt_sl = 0 # m
alt_1 = 1000 # m
alt_2 = 5000 # m
alt_3 = 10000 # m
rho_sl = float(np.squeeze(Atmosphere(alt_sl).density))
rho_alt1 = float(np.squeeze(Atmosphere(alt_1).density))
rho_alt2 = float(np.squeeze(Atmosphere(alt_2).density))
rho_alt3 = float(np.squeeze(Atmosphere(alt_3).density))


### --- Plot Drag Polar and Max CL/CD --- ###
CL_vector = np.linspace(min(CL) - .2, max(CL) + .2, int(1e3))
CD_vector_cambered = CD0 + K * (CL_vector - CL0)**2
CD_vector_symmetric = CD0 + K * CL_vector**2
plt.figure()
plt.plot(CD, CL, "x", label="data")
plt.plot(CD_vector_symmetric, CL_vector, label="symmetric model")
plt.plot(CD_vector_cambered, CL_vector, label="cambered model")
plt.xlabel('$C_D$')
plt.ylabel('$C_L$')
plt.title("Drag Polar of Cessna 172S")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()

CL_mind_cambered = np.sqrt(CD0/K + CL0**2)
CL_mind_symmetric = np.sqrt(CD0/K)
CD_mind_symmetric = CD0 + K*CL_mind_symmetric**2
CD_mind_cambered = CD0 + K*(CL_mind_symmetric - CL0)**2
CL_over_CD_cambered = CL_vector/CD_vector_cambered
CL_over_CD_symmetric = CL_vector/CD_vector_symmetric
plt.figure()
plt.plot(CL_vector, CL_over_CD_cambered, label="cambered")
plt.axvline(x=CL_mind_cambered, color="r", linestyle="--", label="CL_mind_cambered")
plt.plot(CL_vector, CL_over_CD_symmetric, label="symmetric")
plt.axvline(x=CL_mind_symmetric, color="r", linestyle="--", label="CL_mind_symmetric")
plt.xlabel('$C_L$')
plt.ylabel('$C_L/C_D$')
plt.title("Aerodynamic Efficiency of Cessna 172S")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()
print(f"Maximum CL/CD is {CL_mind_cambered/CD_mind_cambered}")


### --- Thrust Required --- ###
rho_vector = [rho_sl, rho_alt1, rho_alt2, rho_alt3]
Vs_vectors = []
Drag_vectors = []
V_md_vector = []
Drag_min = []

for rho in rho_vector:

    # Determine stall speed
    Vstall = np.sqrt(W / (0.5 * rho * S * CLmax))

    # Determine A and B
    A = CD0 * 0.5 * rho * S
    B = K * W ** 2 / 0.5 / rho / S

    # Flight speed vector
    Vs = np.linspace(Vstall, 150, 1000)

    # Define drags
    Dind = B * Vs**-2
    Dprof = A * Vs**2
    D = Dind + Dprof

    # Get minimum drag
    Vmd = (B/A)**.25
    md = A * Vmd**2 + B * Vmd**-2

    # Store 
    Vs_vectors.append(Vs)
    Drag_vectors.append(D)
    V_md_vector.append(Vmd)
    Drag_min.append(md)

plt.figure()
plt.plot(Vs_vectors[0], Drag_vectors[0], label="Altitude = " + str(alt_sl) + " m")
plt.plot(Vs_vectors[1], Drag_vectors[1], label="Altitude = " + str(alt_1) + " m")
plt.plot(Vs_vectors[2], Drag_vectors[2], label="Altitude = " + str(alt_2) + " m")
plt.plot(Vs_vectors[3], Drag_vectors[3], label="Altitude = " + str(alt_3) + " m")
plt.plot(V_md_vector, Drag_min, "x")
plt.xlabel('$TAS [m/s]$')
plt.ylabel('$Drag [N]$')
plt.title("Drag vs. TAS at Various Altitudes")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()
plt.show()


### --- Power Available and Power Required --- ### 
P_vectors = []
V_mp_vector = []
P_min = []
P_A_vector = []

for rho in rho_vector:

    # Determine density ratio
    sigma = rho/rho_sl

    # Determine stall speed
    Vstall = np.sqrt(W / (0.5 * rho * S * CLmax))

    # Determine A and B
    A = CD0 * 0.5 * rho * S
    B = K * W ** 2 / 0.5 / rho / S

    # Flight speed vector
    Vs = np.linspace(Vstall, 150, 1000)

    # Define power
    Pind = B * Vs**-1
    Pprof = A * Vs**3
    P = (Pind + Pprof) / 1000

    # Define available power at altitude
    P_A_alt = P_SL * sigma

    # Get minimum power and mp speed
    Vmp = (B/3/A)**.25
    mp = (A * Vmp**3 + B * Vmp**-1) / 1000

    # Store 
    P_vectors.append(P)
    V_mp_vector.append(Vmp)
    P_min.append(mp)
    P_A_vector.append(P_A_alt)

plt.figure()
colors = plt.cm.tab10(range(len(rho_vector)))
for idx, (vs, p_req, alt, p_avail) in enumerate(zip(Vs_vectors, P_vectors,
                                                   [alt_sl, alt_1, alt_2, alt_3], P_A_vector)):
    plt.plot(vs, p_req, label=f"Altitude = {alt} m", color=colors[idx])
    plt.plot(vs, np.full_like(vs, p_avail), "--", color=colors[idx], alpha=0.7,
             label=f"P_A at {alt} m")
    plt.plot(V_mp_vector[idx], P_min[idx], "x", color=colors[idx])
plt.xlabel('$TAS [m/s]$')
plt.ylabel('$Power [kW]$')
plt.title("Power vs. TAS at Various Altitudes")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()


### --- Maximum Range --- ###
# Determine A and B
A = CD0 * 0.5 * rho * S
B = K * W ** 2 / 0.5 / rho / S

# Determine minimum drag airspeed
Vmd = (B/A)**.25

# Cruise Climb Range Starting from an alt_2
CL_cruise = (2 * W) / (rho_alt2 * S * Vmd **2)
CD_cruise = CD0 + K*CL_cruise**2
CL_over_CD_max = CL_cruise / CD_cruise
R_max_constant_airspeed = (eta_prop / (SFC * g)) * CL_over_CD_max * math.log(MTOM/m_final)
print(R_max_constant_airspeed/1000)


### --- Flight Envelope --- ### 
# Initialize the altitude vector in meters
alt_vector = np.linspace(0, 10000)

# Initialize the speed vector for ceiling calculation
v_vect = np.linspace(1, 200) 

# Loop through the altitude vector to find maximum rate of climb at each altitude
R_C_max = []
for alt in alt_vector:
    rho_sl = float(np.squeeze(Atmosphere(0).density))
    rho = float(np.squeeze(Atmosphere(alt).density))
    sigma = rho/rho_sl
    P_A_alt = P_SL * 1000 * sigma
    P_A_vect = np.full(v_vect.shape, P_A_alt)

    # Determine A and B
    A = CD0 * 0.5 * rho * S
    B = K * W ** 2 / 0.5 / rho / S

    # Determine power required at altitude
    Pind = B * v_vect**-1
    Pprof = A * v_vect**3
    P_R_vect = (Pind + Pprof) / 1000

    # Determine service ceiling
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
    P_available = P_SL * 1000 * sigma
    V_stall = math.sqrt(2*W/(rho*S*CLmax))

    A = 0.5 * rho * S * CD0 
    B = (2 * K * W**2) / (rho * S)

    c4 = A
    c3 = 0.0
    c2 = 0
    c1 = -P_available
    c0 = B

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





