import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import aerosandbox as asb
import aerosandbox.numpy as np
from EVTOLMDO.Optimization import create_aircraft_geometry
from tabulate import tabulate
import matplotlib.pyplot as plt


# Fixed parameters
S_wing = 0.116
fuselage_l = 1
AR_wing = 8
taper_wing=0.8
theta_wing=0*(math.pi/180)
tail_arm_perc=0.4
AR_hstab=4
taper_hstab=0.8
c_VT=0.05
AR_vstab=2.27 
taper_vstab=0.6
eta_t = 0.9
static_margin = 0.15

output = create_aircraft_geometry(
        S_wing = S_wing, 
        AR_wing = AR_wing, 
        taper_wing =  taper_wing, 
        theta_wing = theta_wing, 
        eta_t=eta_t, 
        static_margin=static_margin,
        tail_arm_perc=tail_arm_perc, 
        AR_hstab=AR_hstab, 
        taper_hstab=taper_hstab, 
        c_VT=c_VT, 
        AR_vstab=AR_vstab, 
        taper_vstab=taper_vstab,
        fuselage_l=fuselage_l
    )    

airplane = output[0]
wing_params = output[1]
hstab_params = output[2]
vstab_params = output[3]


def print_table(d: dict, title: str):
    print(f"\n{title}")
    print(tabulate([(k, v) for k, v in d.items()],
                   headers=["Parameter", "Value"],
                   tablefmt="github"))

print_table(wing_params, "Main Wing Parameters")
print_table(hstab_params, "Horizontal Stabilizer Parameters")
print_table(vstab_params, "Vertical Stabilizer Parameters")
airplane.draw_three_view()

avl = True

if not avl: 
    # Run VLM
    alpha = np.arange(0,2,1)
    CL_vlm = []
    CD_vlm = []
    CL_over_CD_vlm = []
    V_cruise = 25

    for i in alpha:
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                velocity=V_cruise,  # m/s
                alpha=i,  # degree
            )
        )

        vlm_output = vlm.run() 

        CL_cruise_vlm = vlm_output["CL"]
        CL_vlm.append(CL_cruise_vlm)

        CD_cruise_vlm = vlm_output["CD"]
        CD_vlm.append(CD_cruise_vlm)

        CL_over_CD_vlm.append(CL_cruise_vlm / CD_cruise_vlm)

    plt.figure()
    plt.plot(CD_vlm, CL_over_CD_vlm)
    plt.show()

else: 
    # Run AVL 
    alpha = np.arange(0,2,1)
    CL_avl = []
    CD_avl = []
    CL_over_CD_avl = []
    V_cruise = 25

    for i in alpha:
        vlm = asb.VortexLatticeMethod(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                velocity=V_cruise,  # m/s
                alpha=i,  # degree
            )
        )

        avl = asb.AVL(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                atmosphere=asb.Atmosphere(altitude=0),
                velocity=V_cruise,
                alpha=i,
                beta=0,
            ),
            avl_command=r"C:\Users\Berk\OneDrive\Desktop\AVL\avl.exe",
        )

        avl_output = avl.run()

        CL_cruise_avl = avl_output["CL"]
        CL_avl.append(CL_cruise_avl)

        CD_cruise_avl = avl_output["CD"]
        CD_avl.append(CD_cruise_avl)

        CL_over_CD_avl.append(CL_cruise_avl / CD_cruise_avl)

    plt.figure()
    plt.plot(CD_avl, CL_over_CD_avl)
    plt.show()