def avl_run(input_dict):

    # --- 0) Imports ---
    import aerosandbox as asb
    import numpy as np

    airplane_object = input_dict["GEOMETRY__object"]

    # --- 1) Define Analysis Points ---
    V_cruise = 25
    alpha = 0

    # --- 2) Run AVL ---
    avl = asb.AVL(
            airplane=airplane_object,
            op_point=asb.OperatingPoint(
                atmosphere=asb.Atmosphere(altitude=0),
                velocity=V_cruise,
                    alpha=alpha,
                    beta=0,
        ),
            avl_command=r"C:\Users\Berk\OneDrive\Desktop\AVL\avl.exe",
    )

    op_point_output = avl.run()

    return op_point_output
