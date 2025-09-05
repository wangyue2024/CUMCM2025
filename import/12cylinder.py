import numpy as np
from typing import List, Tuple, Dict

# =============================================================================
# SECTION 1: CORE ANALYTICAL ENGINE & HELPERS
# =============================================================================

def _get_all_polynomial_coefficients(
    m0: np.ndarray, 
    vm: np.ndarray, 
    p_target: np.ndarray, 
    p_det: np.ndarray, 
    t_det: float, 
    v_sink: np.ndarray, 
    r_smoke: float
) -> Dict[str, np.ndarray]:
    """
    一次性计算所有需要的多项式系数。
    """
    L0 = p_target - m0
    L1 = -vm
    S0 = p_det - v_sink * t_det - m0
    S1 = v_sink - vm
    
    d2 = np.dot(L1, L1)
    d1 = 2 * np.dot(L0, L1)
    d0 = np.dot(L0, L0)
    D_coeffs = np.array([d2, d1, d0])
    
    X2 = np.cross(S1, L1)
    X1 = np.cross(S0, L1) + np.cross(S1, L0)
    X0 = np.cross(S0, L0)
    
    q4 = np.dot(X2, X2)
    q3 = 2 * np.dot(X2, X1)
    q2 = 2 * np.dot(X2, X0) + np.dot(X1, X1)
    q1 = 2 * np.dot(X1, X0)
    q0 = np.dot(X0, X0)
    Q_coeffs = np.array([q4, q3, q2, q1, q0])
    
    R_sq = r_smoke**2
    P4_coeffs = np.polyadd(Q_coeffs, -R_sq * np.pad(D_coeffs, (len(Q_coeffs) - len(D_coeffs), 0), 'constant'))
    
    p2a_2 = np.dot(S1, S1)
    p2a_1 = 2 * np.dot(S0, S1)
    p2a_0 = np.dot(S0, S0) - R_sq
    P2_A_coeffs = np.array([p2a_2, p2a_1, p2a_0])
    
    n2 = np.dot(S1, L1)
    n1 = np.dot(S0, L1) + np.dot(S1, L0)
    n0 = np.dot(S0, L0)
    N_coeffs = np.array([n2, n1, n0])
    
    return {'P4_coeffs': P4_coeffs, 'P2_A_coeffs': P2_A_coeffs, 'N_coeffs': N_coeffs}

def _solve_polynomial_inequality_in_interval(coeffs: np.ndarray, interval: Tuple[float, float]) -> float:
    """
    求解多项式 P(t) <= 0 在指定区间内的解的总时长。
    """
    t_start, t_end = interval
    if t_start >= t_end: return 0.0
    if len(coeffs) == 1: roots = []
    else: roots = np.roots(coeffs)
    
    real_roots_in_interval = sorted([r.real for r in roots if np.isreal(r) and t_start < r.real < t_end])
    points = sorted(list(set([t_start, t_end] + real_roots_in_interval)))
    
    total_duration = 0.0
    for i in range(len(points) - 1):
        p_start, p_end = points[i], points[i+1]
        t_mid = (p_start + p_end) / 2.0
        if np.polyval(coeffs, t_mid) <= 0:
            total_duration += (p_end - p_start)
    return total_duration

def _solve_inequality_and_get_intervals(coeffs: np.ndarray, interval: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    求解多项式 P(t) <= 0 并返回一个解区间的列表。
    """
    t_start, t_end = interval
    if t_start >= t_end: return []
    if len(coeffs) == 1: roots = []
    else: roots = np.roots(coeffs)
    
    real_roots_in_interval = sorted([r.real for r in roots if np.isreal(r) and t_start < r.real < t_end])
    points = sorted(list(set([t_start, t_end] + real_roots_in_interval)))
    
    solution_intervals = []
    for i in range(len(points) - 1):
        p_start, p_end = points[i], points[i+1]
        t_mid = (p_start + p_end) / 2.0
        if np.polyval(coeffs, t_mid) <= 0:
            solution_intervals.append((p_start, p_end))
    return solution_intervals

# =============================================================================
# SECTION 2: CALCULATION FUNCTIONS FOR DIFFERENT TARGET TYPES
# =============================================================================

def calculate_shielding_time_robust(
    missile_pos_initial: np.ndarray, missile_velocity: np.ndarray, target_pos: np.ndarray,
    smoke_detonation_pos: np.ndarray, smoke_detonation_time: float, smoke_effective_duration: float,
    smoke_sink_velocity: np.ndarray = np.array([0., 0., -3.]), smoke_radius: float = 10.0
) -> float:
    """
    [FOR POINT TARGET] 鲁棒地计算单个烟幕对视线的总遮蔽时长。
    """
    T_start, T_end = smoke_detonation_time, smoke_detonation_time + smoke_effective_duration
    all_coeffs = _get_all_polynomial_coefficients(
        m0=missile_pos_initial, vm=missile_velocity, p_target=target_pos,
        p_det=smoke_detonation_pos, t_det=smoke_detonation_time,
        v_sink=smoke_sink_velocity, r_smoke=smoke_radius
    )
    N_coeffs = all_coeffs['N_coeffs']
    critical_roots = np.roots(N_coeffs)
    critical_points = sorted([r.real for r in critical_roots if np.isreal(r) and T_start < r.real < T_end])
    boundary_points = sorted(list(set([T_start, T_end] + critical_points)))
    
    total_shielding_time = 0.0
    for i in range(len(boundary_points) - 1):
        start, end = boundary_points[i], boundary_points[i+1]
        t_mid = (start + end) / 2.0
        n_val_mid = np.polyval(N_coeffs, t_mid)
        
        if n_val_mid >= 0: # Case 1
            duration = _solve_polynomial_inequality_in_interval(all_coeffs['P4_coeffs'], (start, end))
        else: # Case 2
            duration = _solve_polynomial_inequality_in_interval(all_coeffs['P2_A_coeffs'], (start, end))
        total_shielding_time += duration
    return total_shielding_time

def calculate_shielding_time_cylinder(
    missile_pos_initial: np.ndarray, missile_velocity: np.ndarray,
    cylinder_center_xy: Tuple[float, float], cylinder_radius: float, cylinder_height: float,
    num_sample_points_on_edge: int, smoke_detonation_pos: np.ndarray,
    smoke_detonation_time: float, smoke_effective_duration: float,
    smoke_sink_velocity: np.ndarray = np.array([0., 0., -3.]), smoke_radius: float = 10.0
) -> float:
    """
    [FOR CYLINDER TARGET] 计算考虑圆柱体目标的总遮蔽时长。
    """
    def _generate_cylinder_sample_points(center_xy, radius, height, num_edge_points):
        points = []
        cx, cy = center_xy
        points.append(np.array([cx, cy, 0.0])); points.append(np.array([cx, cy, height]))
        for i in range(num_edge_points):
            angle = 2 * np.pi * i / num_edge_points
            x, y = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
            points.append(np.array([x, y, 0])); points.append(np.array([x, y, height])); points.append(np.array([x, y, height / 2.0]))
        return points

    def _intersect_two_interval_lists(list_a, list_b):
        intersection = []
        i, j = 0, 0
        while i < len(list_a) and j < len(list_b):
            start_a, end_a = list_a[i]; start_b, end_b = list_b[j]
            overlap_start, overlap_end = max(start_a, start_b), min(end_a, end_b)
            if overlap_start < overlap_end:
                intersection.append((overlap_start, overlap_end))
            if end_a < end_b: i += 1
            else: j += 1
        return intersection

    def get_shielding_intervals_for_point(target_point):
        T_start, T_end = smoke_detonation_time, smoke_detonation_time + smoke_effective_duration
        all_coeffs = _get_all_polynomial_coefficients(
            m0=missile_pos_initial, vm=missile_velocity, p_target=target_point,
            p_det=smoke_detonation_pos, t_det=smoke_detonation_time,
            v_sink=smoke_sink_velocity, r_smoke=smoke_radius
        )
        N_coeffs = all_coeffs['N_coeffs']
        critical_roots = np.roots(N_coeffs)
        critical_points = sorted([r.real for r in critical_roots if np.isreal(r) and T_start < r.real < T_end])
        boundary_points = sorted(list(set([T_start, T_end] + critical_points)))
        
        total_intervals = []
        for i in range(len(boundary_points) - 1):
            start, end = boundary_points[i], boundary_points[i+1]
            t_mid = (start + end) / 2.0
            n_val_mid = np.polyval(N_coeffs, t_mid)
            if n_val_mid >= 0: # Case 1
                intervals = _solve_inequality_and_get_intervals(all_coeffs['P4_coeffs'], (start, end))
            else: # Case 2
                intervals = _solve_inequality_and_get_intervals(all_coeffs['P2_A_coeffs'], (start, end))
            total_intervals.extend(intervals)
        return total_intervals

    sample_points = _generate_cylinder_sample_points(cylinder_center_xy, cylinder_radius, cylinder_height, num_sample_points_on_edge)
    if not sample_points: return 0.0

    common_intervals = get_shielding_intervals_for_point(sample_points[0])
    for i in range(1, len(sample_points)):
        if not common_intervals: return 0.0
        intervals_for_current_point = get_shielding_intervals_for_point(sample_points[i])
        common_intervals = _intersect_two_interval_lists(common_intervals, intervals_for_current_point)

    return sum(end - start for start, end in common_intervals)

# =============================================================================
# MAIN EXECUTION: SOLVING PROBLEM 1 WITH BOTH MODELS FOR COMPARISON
# =============================================================================
if __name__ == '__main__':
    # --- Problem 1 Constants and Setup ---
    G = 9.8
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    
    CYLINDER_CENTER_XY = (0.0, 200.0)
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0
    NUM_SAMPLES_EDGE = 12
    
    M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
    V_MISSILE_SPEED = 300.0
    direction_vec_m1 = P_FALSE_TARGET - M1_INITIAL_POS
    u_missile_m1 = direction_vec_m1 / np.linalg.norm(direction_vec_m1)
    V_VEC_MISSILE_M1 = V_MISSILE_SPEED * u_missile_m1
    
    FY1_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])
    v_uav_speed = 120.0
    uav_direction = (P_FALSE_TARGET[:2] - FY1_INITIAL_POS[:2])
    uav_direction /= np.linalg.norm(uav_direction)
    v_vec_uav = np.array([uav_direction[0], uav_direction[1], 0]) * v_uav_speed
    
    t_launch = 1.5
    dt_det = 3.6
    
    p_launch = FY1_INITIAL_POS + v_vec_uav * t_launch
    v_launch = v_vec_uav
    p_detonation = p_launch + v_launch * dt_det + 0.5 * np.array([0, 0, -G]) * dt_det**2
    t_detonation = t_launch + dt_det
    
    print("--- Calculating Shielding Time for Problem 1 ---")
    print(f"Scenario: Fixed UAV flight, single smoke bomb.")
    print("-" * 50)
    
    # --- Calculation for Simplified Point Target ---
    print("Method 1: Simplified Point Target Model")
    # The single-point function is now defined in this script, so we can call it directly.
    point_target_time = calculate_shielding_time_robust(
        missile_pos_initial=M1_INITIAL_POS,
        missile_velocity=V_VEC_MISSILE_M1,
        target_pos=np.array([CYLINDER_CENTER_XY[0], CYLINDER_CENTER_XY[1], 0.0]), # Bottom center
        smoke_detonation_pos=p_detonation,
        smoke_detonation_time=t_detonation,
        smoke_effective_duration=20.0
    )
    print(f"Total Shielding Time (Point Target): {point_target_time:.8f} s")
    
    # --- Calculation for Full Cylinder Target ---
    print("\nMethod 2: Full Cylinder Target Model")
    print(f"Cylinder sampling: {2 + 3 * NUM_SAMPLES_EDGE} points")
    cylinder_target_time = calculate_shielding_time_cylinder(
        missile_pos_initial=M1_INITIAL_POS,
        missile_velocity=V_VEC_MISSILE_M1,
        cylinder_center_xy=CYLINDER_CENTER_XY,
        cylinder_radius=CYLINDER_RADIUS,
        cylinder_height=CYLINDER_HEIGHT,
        num_sample_points_on_edge=NUM_SAMPLES_EDGE,
        smoke_detonation_pos=p_detonation,
        smoke_detonation_time=t_detonation,
        smoke_effective_duration=20.0
    )
    print(f"Total Shielding Time (Cylinder Target): {cylinder_target_time:.8f} s")
    
    print("-" * 50)
    error = point_target_time - cylinder_target_time
    if point_target_time > 0:
        relative_error = (error / point_target_time) * 100
        print(f"Difference (Point - Cylinder): {error:.8f} s")
        print(f"Relative Overestimation by Point Model: {relative_error:.2f}%")
    else:
        print("No shielding time found by either model.")