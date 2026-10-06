from examples.small_UAV_lift_cruise.design_campaign import generate_design_campaign
from libraries.aerospace.optimization.small_GH2_UAV_eVTOL_MOO import small_UAV_eVTOL_MOO
from libraries.aerospace.geometry.airframe_fw import create_aircraft_geometry
from pathlib import Path
import math


repo_root = Path(__file__).resolve().parents[1]
yaml_path = repo_root / "templates" / "aerospace" / "small_UAV_lift_cruise.yaml"


campaign = generate_design_campaign(yaml_path)

# --- Create inputs ---
# Read bounds
global_bounds = {}
for parameter in campaign.global_parameters.known_variables + campaign.global_parameters.optimization_variables + campaign.global_parameters.computed_variables:
    if not parameter.bounds == None:
        global_bounds[parameter.tag + "_bounds"] = parameter.bounds

component_bounds = {}
for component in campaign.components:
    for parameter in component.known_variables + component.optimization_variables + component.computed_variables :
        if not parameter.bounds == None:
            component_bounds[parameter.tag + "_bounds"] = parameter.bounds


# Read known variables
known_global_variables = {}
for parameter in campaign.global_parameters.known_variables:
    known_global_variables[parameter.tag] = parameter.value

known_component_variables = {}
for component in campaign.components:
    for parameter in component.known_variables:
        known_component_variables[parameter.tag] = parameter.value

known_scenario_variables = {}
for scenario in campaign.scenarios:
    for parameter in scenario.known_variables:
        known_scenario_variables[parameter.tag] = parameter.value

# Create input dictionary
bounds = {**global_bounds, **component_bounds}
known_variables = {**known_global_variables, **known_component_variables, **known_scenario_variables}
input_dict = {**known_variables, **bounds}

# Run MOO
output = small_UAV_eVTOL_MOO(input_dict)

# Update computed and optimization variables
for parameter in campaign.global_parameters.computed_variables:
    if parameter.tag in output:
        parameter.value = output[parameter.tag]

for parameter in campaign.global_parameters.optimization_variables:
    if parameter.tag in output:
        parameter.value = output[parameter.tag]

for component in campaign.components:
    for parameter in component.computed_variables:
        if parameter.tag in output:
            parameter.value = output[parameter.tag]

for component in campaign.components:
    for parameter in component.optimization_variables:
        if parameter.tag in output:
            parameter.value = output[parameter.tag]

for scenario in campaign.scenarios:
    for parameter in scenario.computed_variables:
        if parameter.tag in output:
            parameter.value = output[parameter.tag]

# Create inputs for geometry function
component_values = {}
for component in campaign.components:
    for parameter in component.known_variables + component.optimization_variables + component.computed_variables:
        component_values[parameter.tag] = parameter.value

input_geometry = {
    "S_wing": component_values.get("S_wing"),
    "AR_wing": component_values.get("AR_wing"),
    "static_margin": component_values.get("static_margin_hstab"),
    "taper_wing": component_values.get("taper_ratio_wing"),
    "theta_wing": math.radians(component_values["theta_wing"]) if component_values.get("theta_wing") is not None else None,
    "tail_arm_perc": component_values.get("tail_arm_perc_hstab"),
    "eta_t": component_values.get("eta_t_hstab"),
    "AR_hstab": component_values.get("AR_hstab"),
    "taper_hstab": component_values.get("taper_ratio_hstab"),
    "c_VT": component_values.get("c_vstab"),
    "AR_vstab": component_values.get("AR_vstab"),
    "taper_vstab": component_values.get("taper_ratio_vstab"),
    "fuselage_l": component_values.get("length_fuselage"),
    "aileron_hinge": component_values.get("aileron_hinge"),
    "aileron_start": component_values.get("aileron_start"),
    "aileron_end": component_values.get("aileron_end"),
    "elevator_hinge": component_values.get("elevator_hinge"),
    "elevator_start": component_values.get("elevator_start"),
    "elevator_end": component_values.get("elevator_end"),
}

missing = [key for key, value in input_geometry.items() if value is None]
if missing:
    raise KeyError(f"Missing geometry inputs: {missing}")

aircraft, geometric_parameters = create_aircraft_geometry(input_geometry)

aircraft.draw_three_view()

