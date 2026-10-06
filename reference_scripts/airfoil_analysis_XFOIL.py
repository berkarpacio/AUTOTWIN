import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.pretty_plots import plt
from tests.test_flow_conditions import flow_conditions
from aerosandbox.tools.pretty_plots import plt, show_plot, set_ticks

XFOIL_COMMAND = r"C:\Users\Berk\OneDrive\Desktop\XFOIL\XFOIL.exe"

# Define flow conditions
h = 90
V = 16
c_ref = 1
Mach, Re = flow_conditions(h=h, c_ref=c_ref, V=V)

# Define airfoils
airfoil_1 = asb.Airfoil("dae51")  # Geometry will be automatically pulled from UIUC database (local)
airfoil_2 = asb.Airfoil("naca4410")
airfoil_3 = asb.Airfoil("naca6409")
airfoil_4 = asb.Airfoil("naca0012")
airfoil_5 = asb.Airfoil("naca2412")
airfoil_6 = asb.Airfoil("clarky")

# Draw one for curiosity
fig, ax = plt.subplots()
airfoil_6.draw(show=False)
set_ticks(0.1, 0.05, 0.1, 0.05)
show_plot()

# List of airfoils
airfoils_list = [airfoil_1, airfoil_2, airfoil_3, airfoil_4, airfoil_5, airfoil_6]

# Analyze airfoils
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for airfoil in airfoils_list:

    analysis = asb.XFoil(
        airfoil=airfoil,
        Re=Re,
        xfoil_command=XFOIL_COMMAND,
    )

    sweep = analysis.alpha(alpha=np.linspace(-2, 15, 18))

    line, = ax[0].plot(sweep['alpha'], sweep['CL'], '.-', label=airfoil.name)
    ax[1].plot(sweep['CD'], sweep['CL'], '.-', color=line.get_color(), label=airfoil.name)

ax[0].set_xlabel(r"Angle of Attack $\alpha$ [deg]")
ax[1].set_xlabel(r"Drag Coefficient $C_D$ [-]")
for axis in ax:
    axis.set_ylabel(r"Lift Coefficient $C_L$ [-]")
    axis.grid(True, alpha=0.3)
    axis.legend()

fig.suptitle(f"Airfoil Comparison at Re = {Re}")
fig.tight_layout()

plt.show()
