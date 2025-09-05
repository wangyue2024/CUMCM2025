import numpy as np
import time

# ==============================================================================
# Part 1: config.py的内容 (Configuration)
# ==============================================================================

class Config:
    """
    A class to hold all configuration parameters, acting as a namespace.
    """
    # --- Physical Constants ---
    G = 9.8  # Gravitational acceleration (m/s^2)

    # --- Target Information ---
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    
    # Cylinder Target Properties
    CYLINDER_CENTER_XY = np.array([0.0, 200.0])
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0
    CYLINDER_SAMPLE_POINTS_EDGE = 10 # Number of points to sample on edges

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
    SIMULATION_TIME_STEP = 0.0001 # Using a smaller time step for better accuracy

# ==============================================================================
# Part 2: models/physics_model.py的内容 (Physics Model)
# ==============================================================================

class PhysicsModelWithCylinder:
    """
    Encapsulates the physics simulation, now with a volumetric cylinder target.
    """
    def __init__(self, missile_id, uav_id, config):
        self.config = config
        self.p_missile_0 = self.config.MISSILE_INITIAL_POS[missile_id]
        self.p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        
        # Pre-calculate missile trajectory
        direction_vec_m = self.config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = self.config.V_MISSILE * self.u_missile
        
        dist_to_target = np.linalg.norm(self.config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / self.config.V_MISSILE

        # Pre-calculate target sample points
        self.target_sample_points = self._generate_cylinder_sample_points()
        print(f"Generated {len(self.target_sample_points)} sample points for the cylinder target.")

    def _generate_cylinder_sample_points(self):
        points = []
        cx, cy = self.config.CYLINDER_CENTER_XY
        radius = self.config.CYLINDER_RADIUS
        height = self.config.CYLINDER_HEIGHT
        num_edge_points = self.config.CYLINDER_SAMPLE_POINTS_EDGE
        
        points.append(np.array([cx, cy, 0]))
        points.append(np.array([cx, cy, height]))
        
        for i in range(num_edge_points):
            angle = 2 * np.pi * i / num_edge_points
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            points.append(np.array([x, y, 0]))
            points.append(np.array([x, y, height]))
            points.append(np.array([x, y, height / 2.0]))
            
        return np.array(points)

    def _get_missile_pos(self, t):
        return self.p_missile_0 + self.v_vec_missile * t

    @staticmethod
    def _distance_point_to_segment(point, seg_start, seg_end):
        if np.array_equal(seg_start, seg_end):
            return np.linalg.norm(point - seg_start)
        
        vec_seg = seg_end - seg_start
        vec_point = point - seg_start
        
        dot_product = np.dot(vec_point, vec_seg)
        seg_len_sq = np.dot(vec_seg, vec_seg)
        
        c = dot_product / seg_len_sq
        
        if c < 0: return np.linalg.norm(point - seg_start)
        if c > 1: return np.linalg.norm(point - seg_end)
        
        projection = seg_start + c * vec_seg
        return np.linalg.norm(point - projection)

    def _check_cylinder_shielding(self, p_smoke_center, p_missile):
        for p_target_sample in self.target_sample_points:
            distance = self._distance_point_to_segment(
                p_smoke_center, p_missile, p_target_sample
            )
            if distance > self.config.R_SMOKE:
                return False
        return True

    def calculate_shielding_time(self, uav_speed, uav_theta, launch_times, det_delays):
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        
        smoke_events = []
        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            dx = v_vec_uav[0] * dt_det
            dy = v_vec_uav[1] * dt_det
            dz = -0.5 * self.config.G * dt_det**2
            p_detonation = p_launch + np.array([dx, dy, dz])
            t_detonation = t_launch + dt_det
            t_end_effective = t_detonation + self.config.T_SMOKE_EFFECTIVE
            smoke_events.append({'p_det': p_detonation, 't_det': t_detonation, 't_end': t_end_effective})
            
        if not smoke_events: return 0.0, {}
        
        sim_start_time = min(event['t_det'] for event in smoke_events)
        sim_end_time = min(max(event['t_end'] for event in smoke_events), self.time_to_impact)

        num_steps = int((sim_end_time - sim_start_time) / self.config.SIMULATION_TIME_STEP) + 1
        if num_steps <= 0: return 0.0, {}
        
        shielded_time_slots = np.zeros(num_steps, dtype=bool)
        time_points = np.linspace(sim_start_time, sim_end_time, num_steps)

        for i, t in enumerate(time_points):
            if shielded_time_slots[i]: continue
            p_missile_t = self._get_missile_pos(t)
            
            for event in smoke_events:
                if event['t_det'] <= t < event['t_end']:
                    dt_since_det = t - event['t_det']
                    p_smoke_center_t = event['p_det'] - np.array([0, 0, self.config.V_SMOKE_SINK * dt_since_det])
                    
                    if self._check_cylinder_shielding(p_smoke_center_t, p_missile_t):
                        shielded_time_slots[i] = True
                        break
                        
        total_shielding_time = np.sum(shielded_time_slots) * self.config.SIMULATION_TIME_STEP
        
        details = {"total_shielding_time": total_shielding_time}
        return total_shielding_time, details

# ==============================================================================
# Part 3: Main execution block to solve Problem 1
# ==============================================================================

if __name__ == '__main__':
    print("--- Running Simulation for Problem 1 with CYLINDER Target Model ---")
    start_time = time.time()
    
    # 1. Initialize Configuration and Model
    cfg = Config()
    model_q1 = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config=cfg)
    
    # 2. Define the fixed strategy from Problem 1
    uav_speed_q1 = 120.0
    direction_vec_q1 = cfg.P_FALSE_TARGET - cfg.UAV_INITIAL_POS['FY1']
    uav_theta_q1 = np.arctan2(direction_vec_q1[1], direction_vec_q1[0])
    
    launch_times_q1 = [1.5]
    det_delays_q1 = [3.6]
    
    # 3. Calculate the result
    shielding_time, details = model_q1.calculate_shielding_time(
        uav_speed_q1, uav_theta_q1, launch_times_q1, det_delays_q1
    )
    
    end_time = time.time()
    
    # 4. Print the results
    print("\n--- Input Strategy ---")
    print(f"UAV Speed: {uav_speed_q1} m/s")
    print(f"UAV Direction: {np.rad2deg(uav_theta_q1):.2f} degrees (towards false target)")
    print(f"Launch Time: {launch_times_q1[0]} s")
    print(f"Detonation Delay: {det_delays_q1[0]} s")
    
    print("\n" + "="*40)
    print(f"CALCULATED EFFECTIVE SHIELDING TIME: {shielding_time:.4f} s")
    print("="*40)
    print(f"Computation took: {end_time - start_time:.4f} seconds.")