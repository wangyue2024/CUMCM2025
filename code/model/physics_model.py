# models/physics_model.py
# Implements the core physics simulation and shielding calculation.

import numpy as np
import config  # Import constants from our config file

class PhysicsModel:
    """
    A class to encapsulate the physics simulation for a single missile-UAV interaction scenario.
    """
    def __init__(self, missile_id, uav_id):
        """
        Initializes the model with specific missile and UAV.
        """
        self.p_missile_0 = config.MISSILE_INITIAL_POS[missile_id]
        self.p_uav_0 = config.UAV_INITIAL_POS[uav_id]
        
        # Pre-calculate missile trajectory parameters
        direction_vec_m = config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = config.V_MISSILE * self.u_missile
        
        # Calculate time to impact for simulation boundary
        dist_to_target = np.linalg.norm(config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / config.V_MISSILE

    def _get_missile_pos(self, t):
        """Calculates missile position at time t."""
        return self.p_missile_0 + self.v_vec_missile * t

    @staticmethod


    def calculate_shielding_time(self, uav_speed, uav_theta, launch_times, det_delays):
        """
        The main cost function. Calculates total shielding time for a given strategy.
        
        Args:
            uav_speed (float): Speed of the UAV (m/s).
            uav_theta (float): Flight direction angle of the UAV in radians (0 is positive x-axis).
            launch_times (list or np.array): A list of launch times for each grenade.
            det_delays (list or np.array): A list of detonation delays for each grenade.
            
        Returns:
            tuple: (total_shielding_time, details_dict)
        """
        # --- 1. Calculate UAV and Smoke Trajectories ---
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        
        smoke_events = []
        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            
            # Projectile motion calculation
            dx = v_vec_uav[0] * dt_det
            dy = v_vec_uav[1] * dt_det
            dz = -0.5 * config.G * dt_det**2
            
            p_detonation = p_launch + np.array([dx, dy, dz])
            
            t_detonation = t_launch + dt_det
            t_end_effective = t_detonation + config.T_SMOKE_EFFECTIVE
            
            smoke_events.append({
                'p_det': p_detonation,
                't_det': t_detonation,
                't_end': t_end_effective,
                'p_launch': p_launch # For visualization
            })
            
        # --- 2. Simulate and Check for Shielding ---
        total_shielding_time = 0
        
        # Determine the simulation time range based on the first and last smoke events
        if not smoke_events:
            return 0.0, {}
        
        sim_start_time = min(event['t_det'] for event in smoke_events)
        sim_end_time = min(max(event['t_end'] for event in smoke_events), self.time_to_impact)

        # Create a boolean array to mark shielded time slots to avoid double counting
        num_steps = int((sim_end_time - sim_start_time) / config.SIMULATION_TIME_STEP) + 1
        shielded_time_slots = np.zeros(num_steps, dtype=bool)
        
        time_points = np.linspace(sim_start_time, sim_end_time, num_steps)

        for i, t in enumerate(time_points):
            if shielded_time_slots[i]:
                continue # Already shielded at this time step

            p_missile_t = self._get_missile_pos(t)
            
            for event in smoke_events:
                if event['t_det'] <= t < event['t_end']:
                    # Calculate current smoke cloud center
                    dt_since_det = t - event['t_det']
                    p_smoke_center_t = event['p_det'] - np.array([0, 0, config.V_SMOKE_SINK * dt_since_det])
                    
                    # Check shielding condition
                    distance = self._distance_point_to_segment(
                        p_smoke_center_t, p_missile_t, config.P_TRUE_TARGET
                    )
                    
                    if distance <= config.R_SMOKE:
                        shielded_time_slots[i] = True
                        break # Move to the next time step once shielded

        total_shielding_time = np.sum(shielded_time_slots) * config.SIMULATION_TIME_STEP

        # --- 3. Prepare Detailed Results for Analysis ---
        details = {
            "uav_speed": uav_speed,
            "uav_theta_deg": np.rad2deg(uav_theta),
            "total_shielding_time": total_shielding_time,
            "smoke_events": smoke_events,
            "missile_trajectory": [self._get_missile_pos(t) for t in time_points],
            "uav_trajectory": [self.p_uav_0 + v_vec_uav * t for t in time_points]
        }
        
        return total_shielding_time, details

# --- Example Usage for Problem 1 ---
if __name__ == '__main__':
    print("--- Running Simulation for Problem 1 ---")
    
    # Initialize the model for M1 and FY1
    model_q1 = PhysicsModel(missile_id='M1', uav_id='FY1')
    
    # Define the fixed strategy from Problem 1
    uav_speed_q1 = 120.0
    # Direction towards false target from FY1's initial position
    direction_vec_q1 = config.P_FALSE_TARGET - config.UAV_INITIAL_POS['FY1']
    uav_theta_q1 = np.arctan2(direction_vec_q1[1], direction_vec_q1[0])
    
    launch_times_q1 = [1.5]
    det_delays_q1 = [3.6]
    
    # Calculate the result
    shielding_time, details = model_q1.calculate_shielding_time(
        uav_speed_q1, uav_theta_q1, launch_times_q1, det_delays_q1
    )
    
    print(f"UAV Speed: {uav_speed_q1} m/s")
    print(f"UAV Direction: {np.rad2deg(uav_theta_q1):.2f} degrees")
    print(f"Launch Time: {launch_times_q1[0]} s")
    print(f"Detonation Delay: {det_delays_q1[0]} s")
    print("\n" + "="*30)
    print(f"Calculated Effective Shielding Time: {shielding_time:.4f} s")
    print("="*30)
    
    # You can now access detailed info for plotting:
    # print(details['smoke_events'][0]['p_launch'])
    # print(details['smoke_events'][0]['p_det'])