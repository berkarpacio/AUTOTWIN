import math
import csv
from pathlib import Path
import aerosandbox as asb
import aerosandbox.numpy as np
from libraries.aerospace.models.geometry.airframe_fw import create_aircraft_geometry
from tabulate import tabulate


# Fixed parameters
input_dict = {}
input_dict["airfoil_main_wing"] = "naca4410"
input_dict["airfoil_hstab"] = "naca4410"
input_dict["airfoil_vstab"] = "naca0012"
input_dict['S_wing'] = 0.115
input_dict['AR_wing'] = 8.0
input_dict['static_margin_hstab'] = 0.28 # (X_NP - X_CG) / mac_wing
input_dict['taper_ratio_wing'] = 0.65
input_dict['theta_wing'] = 3
input_dict['tail_arm_perc_hstab'] = 0.49
input_dict['eta_t_hstab'] = 0.9
input_dict['AR_hstab'] = 4
input_dict['taper_ratio_hstab'] = 0.8
input_dict['c_vstab'] = 0.05
input_dict['AR_vstab'] = 2.27
input_dict['taper_ratio_vstab'] = 0.6
input_dict['length_fuselage'] = 1.0
input_dict['aileron_hinge'] = 0.2
input_dict['aileron_start'] = 0.60
input_dict['aileron_end'] = 0.95
input_dict['elevator_hinge'] = 0.3
input_dict['elevator_start'] = 0.08
input_dict['elevator_end'] = 0.8
input_dict["tail_z_loc"] = 0.14

output = create_aircraft_geometry(input_dict)    

airplane = output[0]
geom_params = output[1]

def print_table(d: dict, title: str):
    print(f"\n{title}")
    print(tabulate([(k, v) for k, v in d.items()],
                   headers=["Parameter", "Value"],
                   tablefmt="github"))

print_table(geom_params, "Main Wing Parameters")

def update_campaign_parameters(params: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return

    headers = rows[0]
    try:
        value_idx = headers.index("value")
    except ValueError:
        raise ValueError("Expected column 'value' not found in campaign parameters CSV.")

    new_col = "ITER 2 - value"
    if new_col in headers:
        new_col_idx = headers.index(new_col)
    else:
        new_col_idx = value_idx + 1
        headers.insert(new_col_idx, new_col)
        for i in range(1, len(rows)):
            rows[i].insert(new_col_idx, "")

    try:
        tag_idx = headers.index("parameter_tag")
    except ValueError:
        raise ValueError("Expected column 'parameter_tag' not found in campaign parameters CSV.")

    for i in range(1, len(rows)):
        row = rows[i]
        if len(row) <= tag_idx:
            continue
        tag = row[tag_idx]
        if tag in params:
            if len(row) <= new_col_idx:
                row.extend([""] * (new_col_idx - len(row) + 1))
            row[new_col_idx] = params[tag]

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows[1:])

campaign_csv_path = Path("examples/small_UAV_lift_cruise/campaign_iterations.csv")
update_campaign_parameters(geom_params, campaign_csv_path)

airplane.draw_three_view()
