def create_aircraft_geometry(input_dict):
    """
    Creates an aircraft geometry as a function of design parameters and outputs geometric parameters of the main wing, 
    horizontal stabilizer, and vertical stabilizer.
    
    Inputs:
    input_dict -> dictionary of aircraft parameters
    
    Outputs:
    airplane -> aerosandbox airplane object
    geometric_parameters -> dictionary of wing, hstab, vstab parameters

    """

    # --- 0) Imports ---
    import math
    import aerosandbox as asb
    import aerosandbox.numpy as np

    
    # --- 1) Read Inputs --- 
    airfoil_main_wing = input_dict["airfoil_main_wing"]
    airfoil_hstab = input_dict["airfoil_hstab"]
    airfoil_vstab = input_dict["airfoil_vstab"]
    S_wing = input_dict['S_wing']
    AR_wing = input_dict['AR_wing']
    static_margin = input_dict['static_margin_hstab']
    taper_wing = input_dict['taper_ratio_wing']
    theta_wing = math.radians(input_dict['theta_wing'])
    tail_arm_perc = input_dict['tail_arm_perc_hstab']
    eta_t = input_dict['eta_t_hstab']
    AR_hstab = input_dict['AR_hstab']
    taper_hstab = input_dict['taper_ratio_hstab']
    c_VT = input_dict['c_vstab']
    AR_vstab = input_dict['AR_vstab']
    taper_vstab = input_dict['taper_ratio_vstab']
    fuselage_l = input_dict['length_fuselage']
    aileron_hinge = input_dict['aileron_hinge']
    aileron_start = input_dict['aileron_start']
    aileron_end = input_dict['aileron_end']
    elevator_hinge = input_dict['elevator_hinge']
    elevator_start = input_dict['elevator_start']
    elevator_end = input_dict['elevator_end']
    tail_z_loc = input_dict["tail_z_loc"]


    ### ---------- Main Wing -----------------------------------------------------------------------------------------

    # Computes
    b_wing = (S_wing * AR_wing) ** 0.5
    c_root_wing = 2 * S_wing / (b_wing * (1 + taper_wing))
    c_tip_wing = c_root_wing * taper_wing
    c_tip_wing_x = (b_wing/2) * np.tan(theta_wing)
    
    mac_wing = (2/3) * c_root_wing * ((1 + taper_wing + taper_wing ** 2)/(1 + taper_wing))
    quarter_chord_wing = mac_wing/4
    mac_wing_y = (b_wing/6) * ((1 + 2*taper_wing)/(1 + taper_wing))
    mac_wing_x = quarter_chord_wing + (mac_wing_y * (c_tip_wing_x))/(b_wing/2)

    aileron_start_y = (b_wing/2) * aileron_start
    aileron_start_x = aileron_start_y * np.tan(theta_wing)
    chord_aileron_start = c_root_wing * (1-(1-taper_wing)*((2*aileron_start_y)/b_wing))
    aileron_end_y = (b_wing/2) * aileron_end
    aileron_end_x = aileron_end_y * np.tan(theta_wing)
    chord_aileron_end = c_root_wing * (1-(1-taper_wing)*((2*aileron_end_y)/b_wing))

    main_wing_parameters = {"b_wing": b_wing, "c_root_wing": c_root_wing, "c_tip_wing": c_tip_wing, "c_tip_wing_x": c_tip_wing_x,
                            "mac_wing": mac_wing, "quarter_chord_wing": quarter_chord_wing, "mac_wing_y": mac_wing_y,
                            "mac_wing_x": mac_wing_x, "aileron_start_y": aileron_start_y, "aileron_start_x": aileron_start_x,
                            "chord_aileron_start": chord_aileron_start, "aileron_end_y": aileron_end_y, "aileron_end_x": aileron_end_x,
                            "chord_aileron_end": chord_aileron_end}
    
    ### ---------- H-stab ----------------------------------------------------------------------------------------------------
    # Computes
    epsilon_alpha = 4 / AR_wing
    c_HT = static_margin / (eta_t * (1 - epsilon_alpha))
    S_hstab = (c_HT * S_wing * mac_wing) / (fuselage_l * tail_arm_perc) # H-Stab area
    b_hstab = (S_hstab * AR_hstab) ** 0.5
    c_root_hstab = 2 * S_hstab / (b_hstab * (1 + taper_hstab))
    c_tip_hstab = c_root_hstab * taper_hstab
    c_tip_hstab_x = c_root_hstab - c_tip_hstab
    theta_hstab = np.atan(c_tip_hstab_x/(b_hstab/2))
    mac_hstab = (2/3) * c_root_hstab * ((1 + taper_hstab + taper_hstab**2)/(1 + taper_hstab))
    quarter_chord_hstab = mac_hstab/4
    mac_hstab_y = (b_hstab/6) * ((1 + 2*taper_hstab)/(1 + taper_hstab))
    mac_hstab_x = quarter_chord_hstab + (mac_hstab_y * c_tip_hstab_x)/(b_hstab/2)
    l_hstab = fuselage_l * tail_arm_perc
    h_stab_translate = mac_wing_x + l_hstab - mac_hstab_x

    elevator_start_y = (b_hstab/2) * elevator_start
    elevator_start_x = elevator_start_y * np.tan(theta_hstab)
    chord_elevator_start = c_root_hstab - elevator_start_x
    elevator_end_y = (b_hstab/2) * elevator_end
    elevator_end_x = elevator_end_y * np.tan(theta_hstab)
    chord_elevator_end = c_root_hstab - elevator_end_x

    hstab_parameters = {"S_hstab": S_hstab, "b_hstab": b_hstab, "c_root_hstab": c_root_hstab, "c_tip_hstab": c_tip_hstab, "c_tip_hstab_x": c_tip_hstab_x,
                            "mac_hstab": mac_hstab, "quarter_chord_hstab": quarter_chord_hstab, "mac_hstab_y": mac_hstab_y,
                            "mac_hstab_x": mac_hstab_x, "l_hstab": l_hstab, "h_stab_translate": h_stab_translate,  "elevator_start_y": elevator_start_y, "elevator_start_x": elevator_start_x,
                            "chord_elevator_start": chord_elevator_start, "elevator_end_y": elevator_end_y, "elevator_end_x": elevator_end_x,
                            "chord_elevator_end": chord_elevator_end, "c_HT": c_HT}

    
    ### ---------- V-stab ---------------------------------------------------------------------------------------------------
    # Computes
    S_vstab = (c_VT * S_wing * b_wing) / (fuselage_l * tail_arm_perc) # V-Stab area
    b_vstab = (S_vstab * AR_vstab) ** 0.5
    c_root_vstab = 2 * S_vstab / (b_vstab * (1 + taper_vstab))
    c_tip_vstab = c_root_vstab * taper_vstab
    c_tip_vstab_x = c_root_vstab - c_tip_vstab
    mac_vstab = (2/3) * c_root_vstab * ((1 + taper_vstab + taper_vstab**2)/(1 + taper_vstab))
    quarter_chord_vstab = mac_vstab/4
    mac_vstab_y = (b_vstab/3) * ((1 + 2*taper_vstab)/(1 + taper_vstab))
    mac_vstab_x = quarter_chord_vstab + (mac_vstab_y * c_tip_vstab_x)/(b_vstab/2)
    l_vstab = fuselage_l * tail_arm_perc
    v_stab_translate = mac_wing_x + l_vstab - mac_vstab_x

    vstab_parameters = {"S_vstab": S_vstab, "b_vstab": b_vstab, "c_root_vstab": c_root_vstab, "c_tip_vstab": c_tip_vstab, 
                        "c_tip_vstab_x": c_tip_vstab_x, "mac_vstab": mac_vstab, "quarter_chord_vstab": quarter_chord_vstab, 
                        "mac_vstab_y": mac_vstab_y, "mac_vstab_x": mac_vstab_x, "l_vstab": l_vstab,
                        "v_stab_translate": v_stab_translate}
    
    ### ---------- Full airplane -----------------------------------------------------------------------------
    airplane = asb.Airplane(
        name="Example Airplane",
        xyz_ref=[mac_wing_x, 0, 0], # should be the CG position
        wings=[
            # Main Wing
            asb.Wing(
                name="Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0,0,0],
                        chord=c_root_wing,
                        twist=0,
                        airfoil=asb.Airfoil(airfoil_main_wing),
                    ),
                    asb.WingXSec(
                        xyz_le=[aileron_start_x, aileron_start_y, 0],
                        chord=chord_aileron_start,
                        twist=0,
                        airfoil=asb.Airfoil(airfoil_main_wing),
                        control_surfaces = [
                            asb.ControlSurface(
                                name="aileron",
                                symmetric=False,
                                deflection=0.0,
                                hinge_point=aileron_hinge,
                                trailing_edge=True
                            )
                        ]
                    ),
                    asb.WingXSec(
                        xyz_le=[aileron_end_x, aileron_end_y, 0],
                        chord=chord_aileron_end,
                        twist=0,
                        airfoil=asb.Airfoil(airfoil_main_wing),
                        control_surfaces = [
                            asb.ControlSurface(
                                name="aileron",
                                symmetric=False,
                                deflection=0.0,
                                hinge_point=aileron_hinge,
                                trailing_edge=True
                            )
                        ]
                    ),
                    asb.WingXSec(
                        xyz_le=[c_tip_wing_x, b_wing/2, 0],
                        chord=c_tip_wing,
                        twist=0,
                        airfoil=asb.Airfoil(airfoil_main_wing),
                    ),
                ]
            ).translate([0,0,0]),
            # H-stab
            asb.Wing(
                name="H-stab",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0,0,0], chord=c_root_hstab, airfoil=asb.Airfoil(airfoil_hstab)),
                    asb.WingXSec(xyz_le=[elevator_start_x, elevator_start_y, 0], chord=chord_elevator_start, airfoil=asb.Airfoil(airfoil_hstab), 
                                 control_surfaces=[asb.ControlSurface(
                                     name = "elevator",
                                     symmetric = True,
                                     deflection = 0.0,
                                     hinge_point = elevator_hinge,
                                     trailing_edge = True )]),
                    asb.WingXSec(xyz_le=[elevator_end_x, elevator_end_y, 0], chord=chord_elevator_end, airfoil=asb.Airfoil(airfoil_hstab), 
                                 control_surfaces=[asb.ControlSurface(
                                     name = "elevator",
                                     symmetric = True,
                                     deflection = 0.0,
                                     hinge_point = elevator_hinge,
                                     trailing_edge = True )]),
                    asb.WingXSec(xyz_le=[c_tip_hstab_x, b_hstab/2, 0], chord=c_tip_hstab, airfoil=asb.Airfoil(airfoil_hstab)),
                ]
            ).translate([h_stab_translate,0,-tail_z_loc]),
            # V-stab
            asb.Wing(
                name="V-stab",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(xyz_le=[0,0,0], chord=c_root_vstab, airfoil=asb.Airfoil(airfoil_vstab)),
                    asb.WingXSec(xyz_le=[c_tip_vstab_x,0,b_vstab/2], chord=c_tip_vstab, airfoil=asb.Airfoil(airfoil_vstab)),
                ]
            ).translate([v_stab_translate,0,-tail_z_loc]),
        ],
    )

    geometric_parameters = {**main_wing_parameters, **hstab_parameters, **vstab_parameters}
    
    return airplane, geometric_parameters