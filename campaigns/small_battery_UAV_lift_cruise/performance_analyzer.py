### This script analyzes the performance of fixed wing aircraft with a power engine ### 

### --- Imports ---
import math
from ambiance import Atmosphere
import numpy as np
import matplotlib.pyplot as plt

### --- Aircraft Parameters --- ###
MTOM = 3.99 # [kg]
V_cruise = 21.47 # [m/s]
eta_pwr = 0.85 * 0.95
g = 9.81
W = MTOM * g
e = 0.85
AR = 9
g = 9.81
P_SL = 315 # sea-level power available [W]
E_batt = 161.5 # [Wh]
S = 0.35 # [m^2]
CD0 = 0.025
alpha = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
CL = np.array([0.32821601113, 0.409714333528, 0.491018327633, 0.572042102997, 0.652663382853, 0.733044598707, 
    0.812855533113, 0.892451693927, 0.97147950705, 1.05018616162, 1.128406977731, 1.206085741268, 
    1.283300465523, 1.360094330278, 1.435713854689, 1.511206227291])
CD = np.array([0.011909470128, 0.016117462346, 0.021262555843, 0.027299860674, 0.034233030945, 0.042029595093, 
    0.050603771444, 0.059979403792, 0.070173605999, 0.081053668293, 0.092613872696, 0.104707092331, 
    0.117587726352, 0.130912220599, 0.144602922276, 0.15899701844]) + CD0
Cm = np.array([-0.141214850398, -0.144874957004, -0.148517969968, -0.152134807522, -0.155599539604, 
    -0.159103452223, -0.162634285771, -0.165903540777, -0.169069846776, -0.172336533463, -0.175664015289,
    -0.179037872737, -0.18217770382, -0.185873177085, -0.187919225247, -0.191113663665])
K = 1 / (math.pi * e * AR)
CL0 = CL[5]
CLmax = max(CL)
alt_sl = 0 # m
alt_1 = 100 # m
alt_2 = 3000 # m
rho_sl = float(np.squeeze(Atmosphere(alt_sl).density))
rho_alt1 = float(np.squeeze(Atmosphere(alt_1).density))
rho_alt2 = float(np.squeeze(Atmosphere(alt_2).density))

### --- Data Plot --- ###
plt.figure()
plt.plot(alpha, CL, "x")
plt.xlabel('$alpha$')
plt.ylabel('$C_L$')
plt.title("Lift Coefficient vs. Alpha")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

plt.figure()
plt.plot(alpha, CL/CD, "x")
plt.xlabel('$alpha$')
plt.ylabel('$C_L/C_D$')
plt.title("Aerodynamic Efficiency")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

plt.figure()
plt.plot(alpha, Cm, "x")
plt.xlabel('$alpha$')
plt.ylabel('$C_M$')
plt.title("Pitching Moment vs. Alpha")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.show()

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
plt.title("Drag Polar of Quadplane")
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
rho_vector = [rho_sl, rho_alt1, rho_alt2]
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
    Vs = np.linspace(Vstall, 50, 1000)

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
    Vs = np.linspace(Vstall, 50, 1000)

    # Define power
    Pind = B * Vs**-1
    Pprof = A * Vs**3
    P = (Pind + Pprof)

    # Define available power at altitude
    P_A_alt = P_SL * sigma

    # Get minimum power and mp speed
    Vmp = (B/3/A)**.25
    mp = (A * Vmp**3 + B * Vmp**-1)

    # Store 
    P_vectors.append(P)
    V_mp_vector.append(Vmp)
    P_min.append(mp)
    P_A_vector.append(P_A_alt)

plt.figure()
colors = plt.cm.tab10(range(len(rho_vector)))
for idx, (vs, p_req, alt, p_avail) in enumerate(zip(Vs_vectors, P_vectors,
                                                   [alt_sl, alt_1, alt_2], P_A_vector)):
    plt.plot(vs, p_req, label=f"Altitude = {alt} m", color=colors[idx])
    plt.plot(vs, np.full_like(vs, p_avail), "--", color=colors[idx], alpha=0.7,
             label=f"P_A at {alt} m")
    plt.plot(V_mp_vector[idx], P_min[idx], "x", color=colors[idx])
plt.xlabel('$TAS [m/s]$')
plt.ylabel('$Power [W]$')
plt.title("Power vs. TAS at Various Altitudes")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()


### --- Range and Endurance --- ###
t_cruise = (E_batt / (P_SL / eta_pwr)) * 60 # minutes
R_cruise = t_cruise * 60 * V_cruise # meters
print(f"Flight time: {t_cruise} minutes")
print(f"Flight range: {R_cruise / 1000} km")

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
    P_A_alt = P_SL * sigma
    P_A_vect = np.full(v_vect.shape, P_A_alt)

    # Determine A and B
    A = CD0 * 0.5 * rho * S
    B = K * W ** 2 / 0.5 / rho / S

    # Determine power required at altitude
    Pind = B * v_vect**-1
    Pprof = A * v_vect**3
    P_R_vect = (Pind + Pprof)

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
    P_available = P_SL * sigma
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
