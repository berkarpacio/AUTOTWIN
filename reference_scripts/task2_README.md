Task 2 — Full Flight Mission Simulation
-------------------------------------------------------------------------------------------------------

What it does
- Simulates a seven-segment mission (takeoff, climb, cruise, loiter, second cruise, powered descent, landing) with longitudinal dynamics equations for a point mass.
- Tracks time, position, altitude, velocity, throttle, mass, and fuel; logs segment transitions.
- Produces a 3x2 subplot figure of key histories across the entire mission.

How to run
1) Install deps: `pip install ambiance numpy matplotlib`.
2) From this folder: `python task2_code.py`.
3) A matplotlib window will show the six subplots for the mission
