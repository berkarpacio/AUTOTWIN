from examples.small_UAV_lift_cruise.design_campaign import generate_design_campaign
from libraries.aerospace.models.optimization.small_GH2_UAV_eVTOL_MOO import small_UAV_eVTOL_MOO
from libraries.aerospace.models.geometry.airframe_fw import create_aircraft_geometry
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
    for parameter in scenario.knonw_variables:
        known_scenario_variables[parameter.tag] = parameter.value

# Create input dictionary
bounds = {**global_bounds, **component_bounds}
known_variables = {**known_global_variables, **known_component_variables, **known_scenario_variables}
input_dict = {**known_variables, **bounds}
print(input_dict)
