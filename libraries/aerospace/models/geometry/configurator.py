from __future__ import annotations

import math
from typing import Any, Dict


def configurator(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an OpenVSP aircraft configuration dictionary from design parameters.

    The returned dictionary is intended to be passed directly to
    libraries.aerospace.models.aerodynamics.vsp_run.vsp_run.
    """

    # Main wing
    S_wing = float(input_dict["S_wing"])
    AR_wing = float(input_dict["AR_wing"])
    taper_wing = float(input_dict["taper_ratio_wing"])
    theta_wing_deg = float(input_dict["theta_wing"])
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
    AR_hstab = float(input_dict["AR_hstab"])
    taper_hstab = float(input_dict["taper_ratio_hstab"])
    theta_hstab = float(input_dict["theta_hstab"])
    c_HT = float(input_dict["c_HT"])
    z_loc_hstab = float(input_dict["z_loc_hstab"])
    x_rot_hstab = float(input_dict["x_rot_hstab"])
    y_rot_hstab = float(input_dict["y_rot_hstab"])
    z_rot_hstab = float(input_dict["z_rot_hstab"])
    root_ThickChord_hstab = float(input_dict["root_ThickChord_hstab"])
    root_Camber_hstab = float(input_dict["root_Camber_hstab"])
    root_CamberLoc_hstab = float(input_dict["root_CamberLoc_hstab"])
    tip_ThickChord_hstab = float(input_dict["tip_ThickChord_hstab"])
    tip_Camber_hstab = float(input_dict["tip_Camber_hstab"])
    tip_CamberLoc_hstab = float(input_dict["tip_CamberLoc_hstab"])
    ctrl_hinge_hstab = float(input_dict["ctrl_hinge_hstab"])
    ctrl_start_hstab = float(input_dict["ctrl_start_hstab"])
    ctrl_end_hstab = float(input_dict["ctrl_end_hstab"])
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

    fuselage_XSec_0_XLoc = float(input_dict["fuselage_XSec_0_XLoc"])
    fuselage_XSec_1_XLoc = float(input_dict["fuselage_XSec_1_XLoc"])
    fuselage_XSec_2_XLoc = float(input_dict["fuselage_XSec_2_XLoc"])
    fuselage_XSec_3_XLoc = float(input_dict["fuselage_XSec_3_XLoc"])
    fuselage_XSec_0_ZLoc = float(input_dict["fuselage_XSec_0_ZLoc"])
    fuselage_XSec_1_ZLoc = float(input_dict["fuselage_XSec_1_ZLoc"])
    fuselage_XSec_2_ZLoc = float(input_dict["fuselage_XSec_2_ZLoc"])
    fuselage_XSec_3_ZLoc = float(input_dict["fuselage_XSec_3_ZLoc"])
    fuselage_XSec_0_w = float(input_dict["fuselage_XSec_0_w"])
    fuselage_XSec_1_w = float(input_dict["fuselage_XSec_1_w"])
    fuselage_XSec_2_w = float(input_dict["fuselage_XSec_2_w"])
    fuselage_XSec_3_w = float(input_dict["fuselage_XSec_3_w"])
    fuselage_XSec_0_h = float(input_dict["fuselage_XSec_0_h"])
    fuselage_XSec_1_h = float(input_dict["fuselage_XSec_1_h"])
    fuselage_XSec_2_h = float(input_dict["fuselage_XSec_2_h"])
    fuselage_XSec_3_h = float(input_dict["fuselage_XSec_3_h"])

    # Geometric relations
    # Edit these for different sizing methodologies
    b_wing = (S_wing * AR_wing) ** 0.5
    c_root_wing = 2.0 * S_wing / (b_wing * (1.0 + taper_wing))
    mac_wing = (2.0 / 3.0) * c_root_wing * ((1.0 + taper_wing + taper_wing**2) / (1.0 + taper_wing))
    quarter_chord_wing = mac_wing / 4.0
    mac_wing_y = (b_wing / 6.0) * ((1.0 + 2.0 * taper_wing) / (1.0 + taper_wing))
    c_tip_wing_x = (b_wing / 2.0) * math.tan(math.radians(theta_wing_deg))
    mac_wing_x = quarter_chord_wing + (mac_wing_y * c_tip_wing_x) / (b_wing / 2.0)
    cg_xyz = (mac_wing_x, 0.0, 0.0)

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

    x_loc_hstab = R_hover + mac_wing + 1.2 * R_hover + tail_rotor_clearance
    y_loc_hstab = R_hover + hover_rotor_clearance + R_cruise
    S_hstab = (c_HT * S_wing * mac_wing) / x_loc_hstab

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
                "root_Camber": root_Camber_hstab,
                "root_CamberLoc": root_CamberLoc_hstab,
                "tip_ThickChord": tip_ThickChord_hstab,
                "tip_Camber": tip_Camber_hstab,
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
    }
