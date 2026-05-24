import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from libraries.aerospace.models.aerodynamics.vsp_run import vsp_run

# --- Geometric relations --- 

# --- Component: Main Wing ---
# parameters 
S_wing = 0.9 # comes from MOO results
AR_wing = 9.9 # comes from MOO results
taper_wing = 0.5 # known variable
theta_wing_deg = 0 # known variable
x_loc_wing = 0 # known variable
y_loc_wing = 0 # known variable
z_loc_wing = 0 # known variable
x_rot_wing = 0 # known variable
y_rot_wing = 0 # known variable
z_rot_wing = 0 # known variable
root_ThickChord_wing = 0.1 # known variable
root_Camber_wing = 0.04 # known variable
root_CamberLoc_wing = 0.4 # known variable
tip_ThickChord_wing = 0.1 # known variable
tip_Camber_wing = 0.04 # known variable
tip_CamberLoc_wing = 0.4 # known variable
ctrl_hinge_wing = 0.3 # known variable
ctrl_start_wing = 0.4 # known variable
ctrl_end_wing = 0.9 # known variable

# --- Component: Tail --- 
# parameters
AR_hstab = 3 # known variable
taper_hstab = 0.5 # known variable
theta_hstab = 2 # known variable
x_rot_hstab = 45 # known variable
c_HT = 0.35 # known variable
z_loc_hstab = 0 # known variable
x_rot_hstab = 45 # known variable
y_rot_hstab = 0 # known variable
z_rot_hstab = 0 # known variable
root_ThickChord_hstab = 0.1 # known variable
root_Camber_hstab = 0.04 # known variable
root_CamberLoc_hstab = 0.4 # known variable
tip_ThickChord_hstab = 0.1 # known variable
tip_Camber_hstab = 0.04 # known variable
tip_CamberLoc_hstab = 0.4 # known variable
ctrl_hinge_hstab = 0.3 # known variable
ctrl_start_hstab = 0.4 # known variable
ctrl_end_hstab = 0.9 # known variable

# --- Component: Fuselage ---
# parameters
tank_l = 0.48 # comes from MOO results
tank_w = 0.23 # comes from MOO results
motor_mount_extension = 0.01 # known variable
hover_rotor_clearance = 0.03 # known variable
fuselage_y_loc = 0 # known variable

# --- Component: Front Right Propeller ---
fr_propeller_num_blades = 2 # known variable
fr_propeller_z_loc = 0.04 # known variable
fr_propeller_x_rot = 45 # known variable
fr_propeller_y_rot = 0 # known variable
fr_propeller_z_rot = 90 # known variable

# --- Component: Front Left Propeller ---
fl_propeller_num_blades = 2 # known variable
fl_propeller_z_loc = 0.04 # known variable
fl_propeller_x_rot = 45 # known variable
fl_propeller_y_rot = 0 # known variable
fl_propeller_z_rot = 90 # known variable

# --- Component: Back Right Propeller ---
br_propeller_num_blades = 2 # known variable
br_propeller_z_loc = 0.04 # known variable
br_propeller_x_rot = 45 # known variable
br_propeller_y_rot = 0 # known variable
br_propeller_z_rot = 90 # known variable

# --- Component: Back Left Propeller --- 
bl_propeller_num_blades = 2 # known variable
bl_propeller_z_loc = 0.04 # known variable
bl_propeller_x_rot = 45 # known variable
bl_propeller_y_rot = 0 # known variable
bl_propeller_z_rot = 90 # known variable

# --- Computes ---
# Main Wing
b_wing = (S_wing * AR_wing) ** 0.5 # computed variable
c_root_wing = 2 * S_wing / (b_wing * (1 + taper_wing)) # computed variable
mac_wing = (2/3) * c_root_wing * ((1 + taper_wing + taper_wing**2) / (1 + taper_wing)) # computed variable
quarter_chord_wing = mac_wing / 4.0 # computed variable
mac_wing_y = (b_wing/6) * ((1 + 2*taper_wing)/(1 + taper_wing)) # computed variable
c_tip_wing_x = (b_wing/2) * math.tan(math.radians(theta_wing_deg)) # computed variable
mac_wing_x = quarter_chord_wing + (mac_wing_y * c_tip_wing_x)/(b_wing/2) # computed variable
cg_xyz = (mac_wing_x, 0.0, 0.0) # computed variable

# Fuselage
fuselage_l = 2.4 * tank_l # computed variable
fuselage_w = 1.2 * tank_w # computed variable
fuselage_x_loc = -1.2*mac_wing # computed variable
fuselage_z_loc = -0.5 * fuselage_w # computed variable
fuselage_XSec_1_XLoc = 0.20 # computed variable
fuselage_XSec_2_XLoc = 0.85 # computed variable
fuselage_XSec_0_w = 0.5 * fuselage_w # computed variable
fuselage_XSec_1_w = fuselage_w # computed variable
fuselage_XSec_2_w = fuselage_w # computed variable
fuselage_XSec_3_w = 0.6 * fuselage_w # computed variable
fuselage_XSec_0_h = 0.6 * fuselage_w # computed variable
fuselage_XSec_1_h = 1.1 * fuselage_w # computed variable
fuselage_XSec_2_h = 1.1 * fuselage_w # computed variable
fuselage_XSec_3_h  = 0.7 * fuselage_w # computed variable


# Cruise Propeller
R_cruise = 0.3 # comes from MOO results

# Front Right Propeller 
R_hover = 0.3 # comes from MOO results
fr_propeller_diameter = 2*R_hover # computed variable
fr_propeller_x_loc = -1.2*R_hover # computed variable
fr_propeller_y_loc = R_cruise + hover_rotor_clearance + R_hover # computed variable

# Front Left Propeller 
R_hover = 0.3 # comes from MOO results
fl_propeller_diameter = 2*R_hover # computed variable
fl_propeller_x_loc = -1.2*R_hover # computed variable
fl_propeller_y_loc = -(R_cruise + hover_rotor_clearance + R_hover) # computed variable

# Back Right Propeller 
R_hover = 0.3 # comes from MOO results
br_propeller_diameter = 2*R_hover # computed variable 
br_propeller_x_loc = 1.2*R_hover + mac_wing # computed variable
br_propeller_y_loc = R_cruise + hover_rotor_clearance + R_hover # computed variable

# Back Left Propeller 
R_hover = 0.3 # comes from MOO results
bl_propeller_diameter = 2*R_hover # computed variable
bl_propeller_x_loc = 1.2*R_hover + mac_wing # computed variable
bl_propeller_y_loc = -(R_cruise + hover_rotor_clearance + R_hover) # computed variable

# Tail
x_loc_hstab = R_hover + mac_wing + 1.2*R_hover + hover_rotor_clearance # computed variable
y_loc_hstab = R_hover + hover_rotor_clearance + R_cruise # computed variable
S_hstab = (c_HT * S_wing * mac_wing) / (x_loc_hstab) # computed variable
b_hstab = (S_hstab * AR_hstab) ** 0.5 # computed variable 
c_root_h = 2 * S_hstab / (b_hstab * (1 + taper_hstab)) # computed variable


input_dict = {
    "cg_xyz": cg_xyz,
    "wings": [
        {
            "name": "Main_Wing",
            "S": S_wing,
            "AR": AR_wing,
            "taper": taper_wing,
            "sweep_deg": theta_wing_deg,
            "x": x_loc_wing,
            "y": y_loc_wing,
            "z": z_loc_wing,
            "x_rot": x_rot_wing,
            "y_rot": y_rot_wing,
            "z_rot": z_rot_wing,
            "root_ThickChord": root_ThickChord_wing,
            "root_Camber": root_Camber_wing,
            "root_CamberLoc": root_CamberLoc_wing,
            "tip_ThickChord": tip_ThickChord_wing,
            "tip_Camber": tip_Camber_wing,
            "tip_CamberLoc": tip_CamberLoc_wing,
            "ctrl_hinge": ctrl_hinge_wing,
            "ctrl_start": ctrl_start_wing,
            "ctrl_end": ctrl_end_wing,
        },
        {
            "name": "H_Stab",
            "S": S_hstab,
            "AR": AR_hstab,
            "taper": taper_hstab,
            "sweep_deg": theta_hstab,
            "x": x_loc_hstab,
            "y": y_loc_hstab,
            "z": z_loc_hstab,
            "x_rot": x_rot_hstab,
            "y_rot": y_rot_hstab,
            "z_rot": z_rot_hstab,
            "root_ThickChord": root_ThickChord_hstab,
            "root_Camber": root_Camber_hstab ,
            "root_CamberLoc": root_CamberLoc_hstab,
            "tip_ThickChord": tip_ThickChord_hstab,
            "tip_Camber":tip_Camber_hstab,
            "tip_CamberLoc": tip_CamberLoc_hstab,
            "ctrl_hinge": ctrl_hinge_hstab,
            "ctrl_start": ctrl_start_hstab,
            "ctrl_end": ctrl_end_hstab,
        },
    ],
    "fuselages": [
        {
            "name": "main_fuselage",
            "length": fuselage_l,

            "fuselage_XLoc": fuselage_x_loc,
            "fuselage_YLoc": fuselage_y_loc,
            "fuselage_ZLoc": fuselage_z_loc,

            "XSec_0_XLoc" : 0.0,
            "XSec_1_XLoc" : fuselage_XSec_1_XLoc,
            "XSec_2_XLoc" : fuselage_XSec_2_XLoc,
            "XSec_3_XLoc" : 1.0,

            "XSec_0_ZLoc" : 0.0,
            "XSec_1_ZLoc" : 0.0,
            "XSec_2_ZLoc" : 0.0,
            "XSec_3_ZLoc" : 0.0,

            "XSec_0_w" : fuselage_XSec_0_w,
            "XSec_1_w" : fuselage_XSec_1_w,
            "XSec_2_w" : fuselage_XSec_2_w,
            "XSec_3_w" : fuselage_XSec_3_w,

            "XSec_0_h" : fuselage_XSec_0_h,
            "XSec_1_h" : fuselage_XSec_1_h,
            "XSec_2_h" : fuselage_XSec_2_h,
            "XSec_3_h" : fuselage_XSec_3_h
        }, 
        {
            "name": "right_motor_boom",
            "length": motor_mount_extension + 2*1.2*R_hover + 2*mac_wing + R_hover,

            "fuselage_XLoc": -(1.2*R_hover + motor_mount_extension),
            "fuselage_YLoc": R_hover + hover_rotor_clearance + R_cruise,
            "fuselage_ZLoc": 0.0,

            "XSec_0_XLoc" : 0.0,
            "XSec_1_XLoc" : 0.5,
            "XSec_2_XLoc" : 1.0,

            "XSec_0_ZLoc" : 0.0,
            "XSec_1_ZLoc" : 0.0,
            "XSec_2_ZLoc" : 0.0,

            "XSec_0_w" : 0.1 * fuselage_w,
            "XSec_1_w" : 0.1 * fuselage_w,
            "XSec_2_w" : 0.1 * fuselage_w,
            
            "XSec_0_h" : 0.1 * fuselage_w,
            "XSec_1_h" : 0.1 * fuselage_w,
            "XSec_2_h" : 0.1 * fuselage_w,

            "RightLAngle": 0.0, 
            "TopLAngle": 0.0, 
            "RightLStrength": 1.0, 
            "TopLStrength": 1.0

        },
         {
            "name": "left_motor_boom",
            "length": motor_mount_extension + 2*1.2*R_hover + 2*mac_wing + R_hover,

            "fuselage_XLoc": -(1.2*R_hover + motor_mount_extension),
            "fuselage_YLoc": -(R_hover + hover_rotor_clearance + R_cruise),
            "fuselage_ZLoc": 0.0,

            "XSec_0_XLoc" : 0.0,
            "XSec_1_XLoc" : 0.5,
            "XSec_2_XLoc" : 1.0,

            "XSec_0_ZLoc" : 0.0,
            "XSec_1_ZLoc" : 0.0,
            "XSec_2_ZLoc" : 0.0,

            "XSec_0_w" : 0.1 * fuselage_w,
            "XSec_1_w" : 0.1 * fuselage_w,
            "XSec_2_w" : 0.1 * fuselage_w,
            
            "XSec_0_h" : 0.1 * fuselage_w,
            "XSec_1_h" : 0.1 * fuselage_w,
            "XSec_2_h" : 0.1 * fuselage_w,

            "RightLAngle": 0.0, 
            "TopLAngle": 0.0, 
            "RightLStrength": 1.0, 
            "TopLStrength": 1.0

        }
    ],
    "props": [
        {
            "name": "hover_FR_prop",
            "diameter": fr_propeller_diameter,
            "num_blades": fr_propeller_num_blades,
            "x": fr_propeller_x_loc,
            "y": fr_propeller_y_loc,
            "z": fr_propeller_z_loc,
            "x_rot": fr_propeller_x_rot,
            "y_rot": fr_propeller_y_rot,
            "z_rot": fr_propeller_z_rot,
        },
        {
            "name": "hover_FL_prop",
            "diameter": fl_propeller_diameter,
            "num_blades": fl_propeller_num_blades,
            "x": fl_propeller_x_loc,
            "y": fl_propeller_y_loc,
            "z": fl_propeller_z_loc,
            "x_rot": fl_propeller_x_rot,
            "y_rot": fl_propeller_y_rot,
            "z_rot": fl_propeller_z_rot,
        },
        {
            "name": "hover_BR_prop",
            "diameter": br_propeller_diameter,
            "num_blades": br_propeller_num_blades,
            "x": br_propeller_x_loc,
            "y": br_propeller_y_loc,
            "z": br_propeller_z_loc,
            "x_rot": br_propeller_x_rot,
            "y_rot": br_propeller_y_rot,
            "z_rot": br_propeller_z_rot,
        },
        {
            "name": "hover_BL_prop",
            "diameter": bl_propeller_diameter,
            "num_blades": bl_propeller_num_blades,
            "x": bl_propeller_x_loc,
            "y": bl_propeller_y_loc,
            "z": bl_propeller_z_loc,
            "x_rot": bl_propeller_x_rot,
            "y_rot": bl_propeller_y_rot,
            "z_rot": bl_propeller_z_rot,
        },
    ]
}

output = vsp_run(input_dict=input_dict)
