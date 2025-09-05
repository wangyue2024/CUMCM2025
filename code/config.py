# config.py
# This file stores all constants and initial conditions for the simulation.

import numpy as np

# --- Physical Constants ---
G = 9.8  # Gravitational acceleration (m/s^2)

# --- Target Information ---
P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
P_TRUE_TARGET = np.array([0.0, 200.0, 0.0])

# --- Missile Information ---
V_MISSILE = 300.0  # Missile speed (m/s)
MISSILE_INITIAL_POS = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}

# --- UAV (Drone) Information ---
V_UAV_MIN = 70.0   # Minimum UAV speed (m/s)
V_UAV_MAX = 140.0  # Maximum UAV speed (m/s)
UAV_INITIAL_POS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}

# --- Smoke Grenade Information ---
R_SMOKE = 10.0            # Effective radius of the smoke cloud (m)
T_SMOKE_EFFECTIVE = 20.0  # Effective duration of the smoke cloud (s)
V_SMOKE_SINK = 3.0        # Sinking speed of the smoke cloud (m/s)
MIN_LAUNCH_INTERVAL = 1.0 # Minimum interval between two launches (s)

# --- Simulation Parameters ---
SIMULATION_TIME_STEP = 0.001 # Time step for discrete simulation (s)
SIMULATION_TIME = 100.0    # Total simulation time (s)