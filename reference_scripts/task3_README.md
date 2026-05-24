Task 3 — Flight Envelope and Service Ceiling
-------------------------------------------------------------------------------------------------------

What it does
- Computes the maximum rate of climb at each altitude and plots the maximum rate of climb vs. altitude. Extracts the service ceiling by interpolating the altitude at which the maximum rate of climb drops to zero.
- Computes the minimum and maximum possible flight speeds across altitudes up to the service ceiling by solving a quartic for power and thrust balance, respecting stall speed.
- Plots altitude vs TAS to get the flight envelope.

How to run
1) Install deps: `pip install ambiance numpy matplotlib`.
2) From this folder: `python task3_code.py`.
3) A matplotlib window will show the maximum rate of climb vs. altitude and altitude vs. TAS curves.