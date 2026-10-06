import math

# UAV annual production volume
Q = 1 # units produced per year []
n = 1 # cumulative aircraft number []

# UAV COTS parameters for Q = 1
COTS_Q1_landed = {
    "hover propeller": {"quantity": 4, "price": 135.45},
    "hover bldc": {"quantity": 4, "price": 279.9},
    "hover ESC": {"quantity": 4, "price": 119.99},
    "cruise propeller": {"quantity": 1, "price": 78.9},
    "cruise bldc": {"quantity": 1, "price": 159},
    "cruise ESC": {"quantity": 1, "price": 109.99},
    "system battery": {"quantity": 1, "price": 888},
    "hybrid battery": {"quantity": 1, "price": 344},
    "fuel cell module": {"quantity": 1, "price": 16287.55},
    "GH2 tank": {"quantity": 1, "price": 3444.57},
    "pressure regulator": {"quantity": 1, "price": 4265.79},
    "PDB": {"quantity": 1, "price": 90},
    "PM02": {"quantity": 1, "price": 18.99},
    "flight controller": {"quantity": 1, "price": 853.7},
    "flight computer": {"quantity": 1, "price": 245.94},
    "servos": {"quantity": 5, "price": 169.99},
    "RC receiver": {"quantity": 1, "price": 27.08},
    "Airspeed sensor": {"quantity": 1, "price": 70.73},
    "IR Lock sensor": {"quantity": 1, "price": 349.42},
    "I2C Splitter": {"quantity": 1, "price": 3.52},
    "Telemetry module": {"quantity": 1, "price": 832.31},
    "GPS sensor": {"quantity": 1, "price": 142.88},
    "payload": {"quantity": 1, "price": 1403.91},
} # [USD]

# Consumables
CONS = 1000 # total consumables cost per UAV [USD]

# 3D-printed airframe manufacturing parameters according to HP JET FUSION 5200 + PA12 GB
p_f_k = 0.05 # probability of print failure []
C_f_k_av = 1000 # average cost consumed by a failed print attempt [USD]
m_p_k = 8 # mass of polymer incorporated into print job [kg]
c_p = 50 # polymer cost per kg [USD/kg]
m_s_k = 0 # mass of support materials per print job [kg]
c_s = 0 # support or sacrificial material price [$/kg]
t_p_k = 10 # total build time per print job [h/job]
E_p_k = 130 # electrical energy consumption per print [kWh]
c_e = 0.1305 # electricity price [$/kWh]
r_l = 15.0 # loaded hourly rate for labor [USD/h]
t_setup_k = 3 # preparation/setup labor time [h]
t_post_k = 20 # support removal, sanding, bonding, insert installation, etc. time [h]
c_post_k = 250 # non-labor post-processing consumables [USD]
P_M = 500000 # printer acquisition price [USD]
V_M = P_M * 0.1 # residual value (10% assumption) [USD]
L_M = 5 # accounting/economic life in years [years]
M_ann = 44000 # annual maintenance [USD/year]
K_ann =  0 # annual calibration and tooling expenses [USD/year]
I_ann = 5000 # other annual machine-specific costs [USD/year]
H_cal = 7008 # theoretically available annual hours per year [hr/year]
u_M = (Q * t_p_k)/H_cal # productive utilization fraction []

# Integration labor
t_l_1 = 120 # first-unit time [h]
S_l = 0.85 # learning curve slope []

# Recoverable rework
p_r = 0.15 # probability of the rework event []
t_rw_r = 8.0 # corrective labor time [h]
t_retest_r = 4.0 # required retest time [h]
C_replacement_r = 250 # replacement material or component cost [USD]

# Variable manufacturing overhead rate
overhead_rate = 0.2 # [%]

# Annual fixed factory cost
F_mfg = 120000 # [USD/year]

# Probability that an assembled production attempt becomes an unrecoverable loss
P_scrap = 0.10 # []


def cost_airframe_three_d_printed(
        p_f_k,
        C_f_k_av,
        m_p_k,
        c_p,
        m_s_k,
        c_s,
        t_p_k,
        E_p_k,
        c_e,
        r_l,
        t_setup_k,
        t_post_k,
        c_post_k,
        P_M,
        V_M,
        L_M,
        M_ann,
        K_ann,
        I_ann,
        H_cal,
        u_M
):

    """ 
    This function computes the total cost of manufacturing a 3D printed 
    airframe, assuming all printed airframe components are nested 
    within a single build job

    p_f_k: probability of print failure
    C_f_k_av: average cost consumed by a failed print attempt
    m_p_k: mass of polymer incorporated into print job
    c_p: polymer cost per kg
    m_s_k: support or sacrificial material mass
    c_s: support material cost per kg
    t_p_k: printer occupation time
    E_p_k: electrical energy consumption per print [kWh]
    c_e: electricity price [$/kWh]
    r_l: loaded hourly rate for labor activity l (manufacturing in this case)
    t_setup_k: preparation/setup labor time
    t_post_k: support removal, sanding, bonding, insert installation, etc. time
    c_post_k: non-labor post-processing consumables
    P_M: printer acquisition price
    V_M: residual value
    L_M: accounting/economic life in years
    M_ann: annual maintenance
    K_ann: annual calibration and tooling expens
    I_ann: other annual machine-specific costs
    H_cal: theoretically available annual hours
    u_M: productive utilization fraction

    """
    # Machine-hour cost
    r_m = ((P_M - V_M)/L_M + M_ann + K_ann + I_ann) / (H_cal * u_M)

    # Cost of one succesful production attempt for print job k
    C_k_0 = ((m_p_k * c_p) + (m_s_k * c_s) + 
             (t_p_k * r_m) + (E_p_k * c_e) + 
             r_l * (t_setup_k + t_post_k) + c_post_k) 
    
    # Total 3D-printed airframe cost
    C_AF = C_k_0 + (p_f_k/(1-p_f_k)) * C_f_k_av

    return C_AF


def cost_labor(
        n,
        r_l,
        t_l_1,
        S_l
):
    
    """ 
    This function computes the total cost associated with the labor
    required for an integration activity. We assume the only activity 
    considered is mechanical assembly.

    n: cumulative aircraft number
    r_l: loaded hourly rate for labor activity l (integration labor in this case)
    t_l_1: first-unit time
    S_l: learning curve slope

    """

    # Compute b_l
    b_l = math.log(S_l)/math.log(2)

    # Compute the total cost of labor assoicated with mechanical assembly
    C_LAB = r_l * t_l_1 * n**b_l

    return C_LAB


def rework_cost(
        p_r,
        t_rw_r,
        t_retest_r,
        C_replacement_r,
        r_l
):
    
    """
    This function computes the total cost associated with a print defect-caused
    rework.

    p_r: probability of the rework event
    t_rw_r: corrective labor time
    t_retest_r: required retest time
    C_replacement_r: replacement material or component cost
    r_l: loaded hourly rate for labor activity l (rework in this case)

    """

    C_RW = p_r * (r_l * (t_rw_r + t_retest_r) + C_replacement_r)

    return C_RW


def cost_subcomponents(dict_of_prices):
    
    C_COTS = 0
    for component in dict_of_prices.values():
        C_COTS += component["price"] * component["quantity"]

    return C_COTS

C_COTS = cost_subcomponents(COTS_Q1_landed)
C_AF = cost_airframe_three_d_printed(p_f_k,
        C_f_k_av,
        m_p_k,
        c_p,
        m_s_k,
        c_s,
        t_p_k,
        E_p_k,
        c_e,
        r_l,
        t_setup_k,
        t_post_k,
        c_post_k,
        P_M,
        V_M,
        L_M,
        M_ann,
        K_ann,
        I_ann,
        H_cal,
        u_M)
C_LAB = cost_labor(
        n,
        r_l,
        t_l_1,
        S_l
)
C_RW = rework_cost(
        p_r,
        t_rw_r,
        t_retest_r,
        C_replacement_r,
        r_l
)

C_OH = overhead_rate * (C_AF + CONS + C_LAB + C_RW) 

unit_cost = (C_COTS + C_AF + CONS + C_LAB + C_RW + C_OH) / (1 - P_scrap) + F_mfg / Q
unit_variable_cost = (C_COTS + C_AF + CONS + C_LAB + C_RW + C_OH) / (1 - P_scrap)
unit_fixed_cost = F_mfg / Q

print(f"Unit UAV cost: {unit_cost} $/UAV")
print("-------- Breakdown of unit cost --------")
print(f"Total variable costs of producing one UAV: {unit_variable_cost} $/UAV")
print(f"Total fixed costs of producing one UAV: {unit_fixed_cost} $/UAV")
print(f"Total cost of COTS components: {C_COTS} $/UAV")
print(f"Total cost of manufacturing a unit: {C_AF} $/UAV")
print(f"Total cost of integration labor per unit: {C_LAB} $/UAV")
print(f"Total cost of consumables per unti: {CONS} $/UAV")
print(f"Total cost of recoverable rework per unit: {C_RW} $/UAV")
print(f"Total cost of manufacturing overhead per unit: {C_OH} $/UAV")


