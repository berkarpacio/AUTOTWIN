def toc_flight(V_cruise, n_D, T_D, T_M, T_T, elec_price, eta_electrolyser, LHV_H2, MF, MWR,
               i, RV, x_ins, m_fuel, indirect_perc, refuel_station_cost, gcs_cost, uav_subcomponents_cost,
               airframe_material_cost, airframe_manufacturing_labor_h, manufacturing_wrap_rate,
               manufacturing_overhead_rate, final_assembly_labor_h, n_uav, n_refuel_stations, 
               n_gcs, support_equipment_cost, uav_life_years, gcs_life_years, support_equipment_life_years,
               station_life_years, water_kg_per_kg_h2, water_price_per_kg, maintenance_parts_per_fh,
               station_maintenance_per_fh, n_operators, launch_recovery_labor_h, operator_monitoring_fraction,
               operator_rate, coverage_efficiency, price_per_km):
    
    """
    Estimates total operating cost per flight for a hydrogen-powered UAS
    surveillance service provider.

    CONOPS assumptions:
    - Company owns the UAV, mobile hydrogen production/refueling station, and GCS.
    - UAV is designed, manufactured, and assembled in-house.
    - Subcomponents are commercially sourced.
    - One operator can assemble, launch, monitor, and recover the UAV.
    - Electrolyzer can produce hydrogen while UAV is flying.
    - Grid electricity is used to produce hydrogen.
    """

    # --- 0) Operational model ---
    cycle_time_h = T_M + T_T
    FC_day = T_D / cycle_time_h # daily operating flights
    FC_annual = n_D * FC_day # annual operating flights
    annual_flight_hours = FC_annual * T_M

    # --- 1) COO per flight ---

    # Annualized ownership cost helper 
    def equivalent_annual_cost(capex, life_years, interest_rate, residual_fraction):
        
        """Equivalent annual cost with residual value."""

        CRF = (
            interest_rate * (1 + interest_rate) ** life_years
            / ((1 + interest_rate) ** life_years - 1)
        )

        SFF = interest_rate / ((1 + interest_rate) ** life_years - 1)

        salvage_value = residual_fraction * capex

        return capex * CRF - salvage_value * SFF

    # UAV unit cost 
    direct_inhouse_uav_cost = (
            uav_subcomponents_cost
            + airframe_material_cost
            + airframe_manufacturing_labor_h * manufacturing_wrap_rate
            + final_assembly_labor_h * manufacturing_wrap_rate
        )
    uav_unit_cost = direct_inhouse_uav_cost * (1 + manufacturing_overhead_rate)
    
    # Total Capex 
    total_uav_capex = n_uav * uav_unit_cost
    total_station_capex = n_refuel_stations * refuel_station_cost
    total_gcs_capex = n_gcs * gcs_cost
    total_support_capex = support_equipment_cost

    total_capex = (
        total_uav_capex
        + total_station_capex
        + total_gcs_capex
        + total_support_capex
    )

    # Total Annualized Capital Cost 
    annualized_uav_cost = equivalent_annual_cost(total_uav_capex, uav_life_years, i, RV)
    annualized_station_cost = equivalent_annual_cost(total_station_capex, station_life_years, i, RV)
    annualized_gcs_cost = equivalent_annual_cost(total_gcs_capex, gcs_life_years, i, RV)
    annualized_support_cost = equivalent_annual_cost(total_support_capex, support_equipment_life_years, i, RV)

    annualized_capital_cost = (
        annualized_uav_cost
        + annualized_station_cost
        + annualized_gcs_cost
        + annualized_support_cost
    )

    # Insurance as annual percentage of insured asset value
    annual_insurance_cost = x_ins * total_capex
    COO_annual = annualized_capital_cost + annual_insurance_cost
    COO_flight = COO_annual / FC_annual

   
    # --- 2) Cash operating cost per flight ---

    # Electricity cost for producing hydrogen
    C_E = (m_fuel * LHV_H2 / eta_electrolyser) * elec_price

    # Water cost, usually very small
    C_W = m_fuel * water_kg_per_kg_h2 * water_price_per_kg

    # UAV maintenance labor + parts
    C_M_uav = T_M * (MF * MWR + maintenance_parts_per_fh)

    # Refueling station maintenance
    C_M_station = T_M * station_maintenance_per_fh

    # Operator cost
    # If operator actively monitors the whole flight, use operator_monitoring_fraction = 1.0.
    # If highly autonomous, reduce this, e.g. 0.25.
    operator_hours = launch_recovery_labor_h + operator_monitoring_fraction * T_M
    C_O = n_operators * operator_rate * operator_hours

    # COC per flight
    COC = C_E + C_W + C_M_uav + C_M_station + C_O


    # --- 3) Indirect operating cost per flight ---
    IOC_flight = indirect_perc * (COC + COO_flight)


    # --- 4) Total operating cost per flight ---
    TOC_flight = COC + COO_flight + IOC_flight
    TOC_annual = TOC_flight * FC_annual
    TOC_hour = TOC_flight / T_M

    # --- 5) TOC processing ---
    km_per_flight = V_cruise * 3.6 * T_M * coverage_efficiency # effective distance covered
    TOC_km = TOC_flight / km_per_flight

    # --- 6) Revenue model ---
    profit_per_flight = (price_per_km * km_per_flight) - TOC_flight
    profit_per_annum = profit_per_flight * FC_annual

    return {
        "h2_per_flight_kg": m_fuel,

        "cycle_time_h": cycle_time_h,
        "flights_per_day": FC_day,
        "flights_per_year": FC_annual,
        "annual_flight_hours": annual_flight_hours,

        "uav_unit_cost_usd": uav_unit_cost,
        "total_capex_usd": total_capex,
        "annualized_capital_cost_usd": annualized_capital_cost,
        "annual_insurance_cost_usd": annual_insurance_cost,

        "electricity_cost_per_flight_usd": C_E,
        "water_cost_per_flight_usd": C_W,
        "uav_maintenance_per_flight_usd": C_M_uav,
        "station_maintenance_per_flight_usd": C_M_station,
        "operator_cost_per_flight_usd": C_O,

        "cash_operating_cost_per_flight_usd": COC,
        "ownership_cost_per_flight_usd": COO_flight,
        "indirect_cost_per_flight_usd": IOC_flight,

        "total_operating_cost_per_flight_usd": TOC_flight,
        "total_operating_cost_per_year_usd": TOC_annual,
        "total_operating_cost_per_hour_usd": TOC_hour,

        "km_per_flight": km_per_flight,
        "total_operating_cost_per_km_usd": TOC_km,

        "profit_per_flight_usd": profit_per_flight,
        "profit_per_annum_usd": profit_per_annum
     }

results = toc_flight(

    n_D = 180, # annual operation days
    T_D = 24, # daily operation hours
    T_M = 8, 
    T_T = 5/60, # turn around time in hours, which is the estimated duration of hydrogen refueling
    elec_price = 0.18, # price of grid electricity in $/kWh
    eta_electrolyser = 0.70, # water electrolysis  efficiency
    LHV_H2 = 33.330, # in kWh/kg
    m_fuel = 140/1000, # mass of fuel in kg
    MF = 0.6, # maintenance man-hours per flight hour ratio
    MWR = 60, # maintenance wrap-rate in USD/hr
    i = 0.03, # interest rate percentage
    RV = 0.05, # percentage residual value of investment
    x_ins = 0.06, # insurance factor
    indirect_perc = 0.2, # we assume indirect cost is 20% of direct operating costs
    price_per_km = 10, # price per km charged to customer

    # Operator model
    n_operators=1,
    operator_rate=45,
    operator_monitoring_fraction=1.0,
    launch_recovery_labor_h=0.25,

    # UAV cost model
    uav_subcomponents_cost=200000.0,
    airframe_material_cost=1000,
    airframe_manufacturing_labor_h=0.0,
    final_assembly_labor_h=0.0,
    manufacturing_wrap_rate=60,
    manufacturing_overhead_rate=0.15,
    uav_life_years=5,

    # Hydrogen production / refueling station
    refuel_station_cost=150000,
    station_life_years=10,
    station_maintenance_per_fh=20,

    # GCS and support equipment
    gcs_cost=8000,
    gcs_life_years=5,
    support_equipment_cost=10000,
    support_equipment_life_years=7,

    # Fleet assumptions
    n_uav=1,
    n_refuel_stations=1,
    n_gcs=1,

    # Mission effectiveness
    V_cruise=28,
    coverage_efficiency = 0.6,

    # Other operating costs
    maintenance_parts_per_fh=60.0,
    water_price_per_kg=0.0,
    water_kg_per_kg_h2=9.0,
)

for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:,.2f}")
    else:
        print(f"{key}: {value}")