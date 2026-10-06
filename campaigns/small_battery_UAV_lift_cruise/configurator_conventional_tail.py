from __future__ import annotations

import math
from typing import Any, Dict


def configurator(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an OpenVSP aircraft configuration dictionary from design parameters.

    The returned dictionary is intended to be passed directly to
    libraries.aerospace.aerodynamics.vsp_run.vsp_run.
    """

    ### --- Read Inputs ---
    # Main wing
    S_wing = float(input_dict["S_wing"])
    AR_wing = float(input_dict["AR_wing"])
    taper_wing = float(input_dict["taper_ratio_wing"])
    theta_wing = float(input_dict["theta_wing"])
    x_loc_wing = float(input_dict["x_loc_wing"])
    y_loc_wing = float(input_dict["y_loc_wing"])
    z_loc_wing = float(input_dict["z_loc_wing"])
    x_rot_wing = float(input_dict["x_rot_wing"])
    y_rot_wing = float(input_dict["y_rot_wing"])
    z_rot_wing = float(input_dict["z_rot_wing"])
    root_ThickChord_wing = float(input_dict["root_ThickChord_wing"])
    root_Camber_wing = float(input_dict["root_Camber_wing"])
    root_CamberLoc_wing = float(input_dict["root_CamberLoc_wing"])
    tip_ThickChord_wing = float(input_dict["tip_ThickChord_wing"])
    tip_Camber_wing = float(input_dict["tip_Camber_wing"])
    tip_CamberLoc_wing = float(input_dict["tip_CamberLoc_wing"])
    ctrl_hinge_wing = float(input_dict["aileron_hinge"])
    ctrl_start_wing = float(input_dict["aileron_start"])
    ctrl_end_wing = float(input_dict["aileron_end"])

    # Tail
    l_moment_fraction = float(input_dict["l_moment_fraction"])
    AR_hstab = float(input_dict["AR_hstab"])
    AR_vstab = float(input_dict["AR_vstab"])
    taper_hstab = float(input_dict["taper_ratio_hstab"])
    taper_vstab = float(input_dict["taper_ratio_vstab"])
    theta_hstab = float(input_dict["theta_hstab"])
    theta_vstab = float(input_dict["theta_vstab"])
    c_HT = float(input_dict["c_HT"])
    c_VT = float(input_dict["c_VT"])
    z_loc_vstab = float(input_dict["z_loc_vstab"])
    x_rot_hstab = float(input_dict["x_rot_hstab"])
    x_rot_vstab = float(input_dict["x_rot_vstab"])
    y_rot_hstab = float(input_dict["y_rot_hstab"])
    y_rot_vstab = float(input_dict["y_rot_vstab"])
    z_rot_hstab = float(input_dict["z_rot_hstab"])
    z_rot_vstab = float(input_dict["z_rot_vstab"])
    root_ThickChord_hstab = float(input_dict["root_ThickChord_hstab"])
    root_ThickChord_vstab = float(input_dict["root_ThickChord_vstab"])
    root_Camber_hstab = float(input_dict["root_Camber_hstab"])
    root_Camber_vstab = float(input_dict["root_Camber_vstab"])
    root_CamberLoc_hstab = float(input_dict["root_CamberLoc_hstab"])
    root_CamberLoc_vstab = float(input_dict["root_CamberLoc_vstab"])
    tip_ThickChord_hstab = float(input_dict["tip_ThickChord_hstab"])
    tip_ThickChord_vstab = float(input_dict["tip_ThickChord_vstab"])
    tip_Camber_hstab = float(input_dict["tip_Camber_hstab"])
    tip_Camber_vstab = float(input_dict["tip_Camber_vstab"])
    tip_CamberLoc_hstab = float(input_dict["tip_CamberLoc_hstab"])
    tip_CamberLoc_vstab = float(input_dict["tip_CamberLoc_vstab"])
    ctrl_hinge_hstab = float(input_dict["ctrl_hinge_hstab"])
    ctrl_hinge_vstab = float(input_dict["ctrl_hinge_vstab"])
    ctrl_start_hstab = float(input_dict["ctrl_start_hstab"])
    ctrl_start_vstab = float(input_dict["ctrl_start_vstab"])
    ctrl_end_hstab = float(input_dict["ctrl_end_hstab"])
    ctrl_end_vstab = float(input_dict["ctrl_end_vstab"])
    tail_rotor_clearance = float(input_dict["tail_rotor_clearance"])

    # Rotor layout
    motor_mount_extension = float(input_dict["motor_mount_extension"])
    hover_rotor_clearance = float(input_dict["hover_rotor_clearance"])
    fuselage_y_loc = float(input_dict["fuselage_y_loc"])
    R_cruise = float(input_dict["radius_cruise_propeller"])
    R_hover = float(input_dict["radius_hover_propeller"])

    # Propellers
    cruise_propeller_num_blades = float(input_dict["cruise_propeller_num_blades"])
    cruise_propeller_offset = float(input_dict["cruise_propeller_offset"])
    cruise_propeller_x_rot = float(input_dict["cruise_propeller_x_rot"])
    cruise_propeller_y_rot = float(input_dict["cruise_propeller_y_rot"])
    cruise_propeller_z_rot = float(input_dict["cruise_propeller_z_rot"])

    fr_propeller_num_blades = float(input_dict["fr_propeller_num_blades"])
    fr_propeller_z_loc = float(input_dict["fr_propeller_z_loc"])
    fr_propeller_x_rot = float(input_dict["fr_propeller_x_rot"])
    fr_propeller_y_rot = float(input_dict["fr_propeller_y_rot"])
    fr_propeller_z_rot = float(input_dict["fr_propeller_z_rot"])

    fl_propeller_num_blades = float(input_dict["fl_propeller_num_blades"])
    fl_propeller_z_loc = float(input_dict["fl_propeller_z_loc"])
    fl_propeller_x_rot = float(input_dict["fl_propeller_x_rot"])
    fl_propeller_y_rot = float(input_dict["fl_propeller_y_rot"])
    fl_propeller_z_rot = float(input_dict["fl_propeller_z_rot"])

    br_propeller_num_blades = float(input_dict["br_propeller_num_blades"])
    br_propeller_z_loc = float(input_dict["br_propeller_z_loc"])
    br_propeller_x_rot = float(input_dict["br_propeller_x_rot"])
    br_propeller_y_rot = float(input_dict["br_propeller_y_rot"])
    br_propeller_z_rot = float(input_dict["br_propeller_z_rot"])

    bl_propeller_num_blades = float(input_dict["bl_propeller_num_blades"])
    bl_propeller_z_loc = float(input_dict["bl_propeller_z_loc"])
    bl_propeller_x_rot = float(input_dict["bl_propeller_x_rot"])
    bl_propeller_y_rot = float(input_dict["bl_propeller_y_rot"])
    bl_propeller_z_rot = float(input_dict["bl_propeller_z_rot"])

    # Fuselage 
    fuselage_l = float(input_dict["fuselage_l"])
    fuselage_w = float(input_dict["fuselage_w"])
    factor = 1 - 1.6/2.2

    fuselage_XSec_0_XLoc = 0.0 * fuselage_l
    fuselage_XSec_1_XLoc = 0.5 * factor
    fuselage_XSec_2_XLoc = 1 - (0.5 * factor)
    fuselage_XSec_3_XLoc = 1.0 
    fuselage_XSec_0_ZLoc = 0.0
    fuselage_XSec_1_ZLoc = 0.0
    fuselage_XSec_2_ZLoc = 0.0
    fuselage_XSec_3_ZLoc = 0.0
    fuselage_XSec_0_w = 0.7 * fuselage_w
    fuselage_XSec_1_w = fuselage_w
    fuselage_XSec_2_w = fuselage_w
    fuselage_XSec_3_w = 0.4 * fuselage_w
    fuselage_XSec_0_h = 0.7 * fuselage_w
    fuselage_XSec_1_h = fuselage_w
    fuselage_XSec_2_h = fuselage_w
    fuselage_XSec_3_h = 0.4 * fuselage_w

    ### --- Geometric relations ---
    # Main Wing
    b_wing = (S_wing * AR_wing) ** 0.5
    c_root_wing = 2.0 * S_wing / (b_wing * (1.0 + taper_wing))
    c_tip_wing = c_root_wing * taper_wing
    c_tip_wing_x = (b_wing / 2.0) * math.tan(math.radians(theta_wing))
    mac_wing = (2.0 / 3.0) * c_root_wing * ((1.0 + taper_wing + taper_wing**2) / (1.0 + taper_wing))
    quarter_chord_wing = mac_wing / 4.0
    mac_wing_y = (b_wing/6.0) * ((1.0 + 2.0 * taper_wing) / (1.0 + taper_wing))
    mac_wing_x = math.tan(math.radians(theta_wing))*mac_wing_y
    cg_xyz = (mac_wing_x + quarter_chord_wing, 0.0, 0.0)

    # Tail
    l_moment = fuselage_l * l_moment_fraction
    # V_stab 
    S_vstab = (c_VT * S_wing * b_wing) / l_moment
    b_vstab = math.sqrt(S_vstab * AR_vstab)
    c_root_vstab = (2*S_vstab)/(b_vstab*(1+taper_vstab))
    c_tip_vstab = c_root_vstab * taper_vstab
    c_tip_vstab_x = (b_vstab / 2.0) * math.tan(math.radians(theta_vstab))
    mac_vstab = (2/3)*(c_root_vstab)*((1+taper_vstab+taper_vstab**2)/(1+taper_vstab))
    quarter_chord_vstab = mac_vstab/4
    mac_vstab_y = (b_vstab/3)*((1+2*taper_vstab)/(1+taper_vstab))
    mac_vstab_x = math.tan(math.radians(theta_vstab))*mac_vstab_y
    x_loc_vstab = (mac_wing_x + quarter_chord_wing + l_moment) - (mac_vstab_x + quarter_chord_vstab)
    y_loc_vstab = 0
    # H_stab
    S_hstab = (c_HT * S_wing * mac_wing) / l_moment
    b_hstab = math.sqrt(S_hstab * AR_hstab)
    c_root_hstab = (2*S_hstab)/(b_hstab*(1+taper_hstab))
    c_tip_hstab = c_root_hstab * taper_hstab
    c_tip_hstab_x = (b_hstab / 2.0) * math.tan(math.radians(theta_hstab))
    mac_hstab = (2/3)*(c_root_hstab)*((1+taper_hstab+taper_hstab**2)/(1+taper_hstab))
    quarter_chord_hstab = mac_hstab/4
    mac_hstab_y = (b_hstab/6)*((1+2*taper_hstab)/(1+taper_hstab))
    mac_hstab_x = math.tan(math.radians(theta_hstab))*mac_hstab_y
    x_loc_hstab = (mac_wing_x + quarter_chord_wing + l_moment) - (mac_hstab_x + quarter_chord_hstab)
    y_loc_hstab = 0
    z_loc_hstab = 0

    # Propellers
    fr_propeller_diameter = 2.0 * R_hover
    fr_propeller_x_loc = -1.2 * R_hover
    fr_propeller_y_loc = R_cruise + hover_rotor_clearance + R_hover
    fl_propeller_diameter = 2.0 * R_hover
    fl_propeller_x_loc = -1.2 * R_hover
    fl_propeller_y_loc = -(R_cruise + hover_rotor_clearance + R_hover)
    br_propeller_diameter = 2.0 * R_hover
    br_propeller_x_loc = 1.2 * R_hover + mac_wing
    br_propeller_y_loc = R_cruise + hover_rotor_clearance + R_hover
    bl_propeller_diameter = 2.0 * R_hover
    bl_propeller_x_loc = 1.2 * R_hover + mac_wing
    bl_propeller_y_loc = -(R_cruise + hover_rotor_clearance + R_hover)


    # Fuselage 
    boom_length = motor_mount_extension + 2.0 * 1.2 * R_hover + 2.0 * mac_wing + R_hover
    boom_x_loc = -(1.2 * R_hover + motor_mount_extension)
    right_boom_y_loc = fr_propeller_y_loc 
    left_boom_y_loc = -right_boom_y_loc
    boom_w = 0.1 * fuselage_w
    fuselage_x_loc = -1.4 * mac_wing
    fuselage_z_loc = -0.5 * fuselage_w

    cruise_propeller_diameter = 2.0 * R_cruise
    cruise_propeller_x_loc = (fuselage_l - abs(fuselage_x_loc)) + cruise_propeller_offset
    cruise_propeller_z_loc = fuselage_z_loc

    return {
        "cg_xyz": cg_xyz,
        "wings": [
            {
                "name": "Main_Wing",
                "S": S_wing,
                "AR": AR_wing,
                "taper": taper_wing,
                "sweep_deg": theta_wing,
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
                "root_Camber": root_Camber_hstab,
                "root_CamberLoc": root_CamberLoc_hstab,
                "tip_ThickChord": tip_ThickChord_hstab,
                "tip_Camber": tip_Camber_hstab,
                "tip_CamberLoc": tip_CamberLoc_hstab,
                "ctrl_hinge": ctrl_hinge_hstab,
                "ctrl_start": ctrl_start_hstab,
                "ctrl_end": ctrl_end_hstab,
            },
            {
                "name": "V_Stab",
                "S": S_vstab,
                "AR": AR_vstab,
                "taper": taper_vstab,
                "sweep_deg": theta_vstab,
                "x": x_loc_vstab,
                "y": y_loc_vstab,
                "z": z_loc_vstab,
                "x_rot": x_rot_vstab,
                "y_rot": y_rot_vstab,
                "z_rot": z_rot_vstab,
                "root_ThickChord": root_ThickChord_vstab,
                "root_Camber": root_Camber_vstab,
                "root_CamberLoc": root_CamberLoc_vstab,
                "tip_ThickChord": tip_ThickChord_vstab,
                "tip_Camber": tip_Camber_vstab,
                "tip_CamberLoc": tip_CamberLoc_vstab,
                "ctrl_hinge": ctrl_hinge_vstab,
                "ctrl_start": ctrl_start_vstab,
                "ctrl_end": ctrl_end_vstab,
            },
        ],
        "fuselages": [
            {
                "name": "main_fuselage",
                "length": fuselage_l,
                "fuselage_XLoc": fuselage_x_loc,
                "fuselage_YLoc": fuselage_y_loc,
                "fuselage_ZLoc": fuselage_z_loc,
                "XSec_0_XLoc": fuselage_XSec_0_XLoc,
                "XSec_1_XLoc": fuselage_XSec_1_XLoc,
                "XSec_2_XLoc": fuselage_XSec_2_XLoc,
                "XSec_3_XLoc": fuselage_XSec_3_XLoc,
                "XSec_0_ZLoc": fuselage_XSec_0_ZLoc,
                "XSec_1_ZLoc": fuselage_XSec_1_ZLoc,
                "XSec_2_ZLoc": fuselage_XSec_2_ZLoc,
                "XSec_3_ZLoc": fuselage_XSec_3_ZLoc,
                "XSec_0_w": fuselage_XSec_0_w,
                "XSec_1_w": fuselage_XSec_1_w,
                "XSec_2_w": fuselage_XSec_2_w,
                "XSec_3_w": fuselage_XSec_3_w,
                "XSec_0_h": fuselage_XSec_0_h,
                "XSec_1_h": fuselage_XSec_1_h,
                "XSec_2_h": fuselage_XSec_2_h,
                "XSec_3_h": fuselage_XSec_3_h,
            },
            {
                "name": "right_motor_boom",
                "length": boom_length,
                "fuselage_XLoc": boom_x_loc,
                "fuselage_YLoc": right_boom_y_loc,
                "fuselage_ZLoc": 0.0,
                "XSec_0_XLoc": 0.0,
                "XSec_1_XLoc": 0.5,
                "XSec_2_XLoc": 1.0,
                "XSec_0_ZLoc": 0.0,
                "XSec_1_ZLoc": 0.0,
                "XSec_2_ZLoc": 0.0,
                "XSec_0_w": boom_w,
                "XSec_1_w": boom_w,
                "XSec_2_w": boom_w,
                "XSec_0_h": boom_w,
                "XSec_1_h": boom_w,
                "XSec_2_h": boom_w,
                "RightLAngle": 0.0,
                "TopLAngle": 0.0,
                "RightLStrength": 1.0,
                "TopLStrength": 1.0,
            },
            {
                "name": "left_motor_boom",
                "length": boom_length,
                "fuselage_XLoc": boom_x_loc,
                "fuselage_YLoc": left_boom_y_loc,
                "fuselage_ZLoc": 0.0,
                "XSec_0_XLoc": 0.0,
                "XSec_1_XLoc": 0.5,
                "XSec_2_XLoc": 1.0,
                "XSec_0_ZLoc": 0.0,
                "XSec_1_ZLoc": 0.0,
                "XSec_2_ZLoc": 0.0,
                "XSec_0_w": boom_w,
                "XSec_1_w": boom_w,
                "XSec_2_w": boom_w,
                "XSec_0_h": boom_w,
                "XSec_1_h": boom_w,
                "XSec_2_h": boom_w,
                "RightLAngle": 0.0,
                "TopLAngle": 0.0,
                "RightLStrength": 1.0,
                "TopLStrength": 1.0,
            },
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
            {
                "name": "cruise_prop",
                "diameter": cruise_propeller_diameter,
                "num_blades": cruise_propeller_num_blades,
                "x": cruise_propeller_x_loc,
                "y": 0,
                "z": cruise_propeller_z_loc,
                "x_rot": cruise_propeller_x_rot,
                "y_rot": cruise_propeller_y_rot,
                "z_rot": cruise_propeller_z_rot,
            }
            
        ],
        
        "S_hstab": S_hstab,
        "S_vstab": S_vstab,
        "x_loc_vstab": x_loc_vstab,
        "x_loc_hstab": x_loc_hstab,
        "y_loc_vstab": y_loc_vstab,
        "z_loc_hstab": z_loc_hstab,
        "b_hstab": b_hstab,
        "b_vstab": b_vstab,
        "c_root_wing": c_root_wing,
        "c_tip_wing": c_tip_wing,
        "c_root_hstab": c_root_hstab,
        "c_tip_hstab": c_tip_hstab,
        "c_root_vstab": c_root_vstab,
        "c_tip_vstab": c_tip_vstab
    }
