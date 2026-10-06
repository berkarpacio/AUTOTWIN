def M_fuel_cell(power_cruise: float, rho_fuel_cell: float): 

    """
    power_cruise -> Cruise power required [W]
    rho_fuel_cell -> Power density of the fuel cell module [W/kg]"""

    mass_fuel_cell = power_cruise / rho_fuel_cell

    return mass_fuel_cell