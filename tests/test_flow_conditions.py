import math

def flow_conditions(h=90, V=16, c_ref=0.190):

    ''' Computes flow properties for aerodynamic analyses
    
    Inputs:
    h: altitude [m]
    V: freestream velocity [m/s]
    c_ref: reference chord length [m]

    Outputs:
    ReCref: Reynold's number
    Mach: Mach number

    '''

    # Convert altitude to feet for NASA atmosphere model
    h_imp = h * 3.28084

    # NASA atmosphere model - English units: https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/atmos.html
    if h_imp >= 82345:
        T = -205.05 + 0.00164 * h_imp
        P = 51.97 * ((T + 459.7) / 389.98)**-11.388

    elif h_imp > 36152:
        T = -70
        P = 473.1 * math.exp(1.73 - 0.000048 * h_imp)

    else:
        T = 59 - 0.00356 * h_imp
        P = 2116 * ((T + 459.7) / 518.6)**5.256

    # Density [slug/ft^3]
    rho_imp = P / (1718 * (T + 459.7))

    # Convert density to kg/m^3
    rho = rho_imp * 515.3788

    # Convert temperature to Kelvin
    T_Kelvin = (T - 32) * (5/9) + 273.15

    # Speed of sound
    gamma = 1.4
    R = 287.05
    a = math.sqrt(gamma * R * T_Kelvin)

    # Dynamic viscosity using Sutherland's law
    mu = (
        1.716e-5
        * (T_Kelvin / 273.15)**(3/2)
        * (273.15 + 110.4)
        / (T_Kelvin + 110.4)
    )

    # Flow conditions
    Mach = V / a
    ReCref = rho * V * c_ref / mu

    return Mach, ReCref, rho


if __name__ == "__main__":
    Mach, ReCref, rho = flow_conditions()
    print(Mach)
    print(ReCref)
    print(rho)