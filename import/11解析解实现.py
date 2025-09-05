import numpy as np
from typing import List, Tuple, Dict

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
    
    返回一个字典，包含:
    - 'P4_coeffs': Case 1 的四次多项式 P(t) = Q(t) - R^2*D(t) 的系数
    - 'P2_A_coeffs': Case 2 的二次多项式 ||Vs(t)||^2 - R^2 的系数
    - 'N_coeffs': 临界点方程 N(t) = 0 的系数
    """
    # 1. 计算 L0, L1, S0, S1 矢量
    L0 = p_target - m0
    L1 = -vm
    S0 = p_det - v_sink * t_det - m0
    S1 = v_sink - vm

    # 2. 计算 D(t) = d2*t^2 + d1*t + d0 的系数
    d2 = np.dot(L1, L1)
    d1 = 2 * np.dot(L0, L1)
    d0 = np.dot(L0, L0)
    D_coeffs = np.array([d2, d1, d0])

    # 3. 计算 V_X(t) = X2*t^2 + X1*t + X0 的矢量系数
    X2 = np.cross(S1, L1)
    X1 = np.cross(S0, L1) + np.cross(S1, L0)
    X0 = np.cross(S0, L0)

    # 4. 计算 Q(t) = q4*t^4 + ... + q0 的系数
    q4 = np.dot(X2, X2)
    q3 = 2 * np.dot(X2, X1)
    q2 = 2 * np.dot(X2, X0) + np.dot(X1, X1)
    q1 = 2 * np.dot(X1, X0)
    q0 = np.dot(X0, X0)
    Q_coeffs = np.array([q4, q3, q2, q1, q0])

    # 5. 计算 Case 1 的最终四次多项式 P(t) 的系数
    R_sq = r_smoke**2
    # 使用 numpy.polyadd 进行多项式相加，需要先对齐阶数
    P4_coeffs = np.polyadd(Q_coeffs, -R_sq * np.pad(D_coeffs, (len(Q_coeffs) - len(D_coeffs), 0), 'constant'))

    # 6. 计算 Case 2 的二次多项式 ||Vs(t)||^2 - R^2 的系数
    # ||Vs(t)||^2 = (S1.S1)t^2 + 2(S0.S1)t + (S0.S0)
    p2a_2 = np.dot(S1, S1)
    p2a_1 = 2 * np.dot(S0, S1)
    p2a_0 = np.dot(S0, S0) - R_sq
    P2_A_coeffs = np.array([p2a_2, p2a_1, p2a_0])

    # 7. 计算临界点方程 N(t) = 0 的系数
    # N(t) = (S1.L1)t^2 + (S0.L1 + S1.L0)t + (S0.L0)
    n2 = np.dot(S1, L1)
    n1 = np.dot(S0, L1) + np.dot(S1, L0)
    n0 = np.dot(S0, L0)
    N_coeffs = np.array([n2, n1, n0])
    
    return {
        'P4_coeffs': P4_coeffs,
        'P2_A_coeffs': P2_A_coeffs,
        'N_coeffs': N_coeffs
    }

def _solve_polynomial_inequality_in_interval(coeffs: np.ndarray, interval: Tuple[float, float]) -> float:
    """
    求解多项式 P(t) <= 0 在指定区间内的解的总时长。
    """
    t_start, t_end = interval
    if t_start >= t_end:
        return 0.0

    # 求解 P(t) = 0 的所有根
    # 对于常数多项式（系数数组只有一个元素），没有根
    if len(coeffs) == 1:
        roots = []
    else:
        roots = np.roots(coeffs)
    
    # 筛选出在区间内的实数根
    real_roots_in_interval = sorted([
        r.real for r in roots if np.isreal(r) and t_start < r.real < t_end
    ])
    
    # 将区间端点和根合并，形成测试点序列
    points = sorted(list(set([t_start, t_end] + real_roots_in_interval)))
    
    total_duration = 0.0
    
    # 遍历由测试点构成的所有子区间
    for i in range(len(points) - 1):
        p_start, p_end = points[i], points[i+1]
        
        # 在子区间中点测试不等式是否成立
        t_mid = (p_start + p_end) / 2.0
        if np.polyval(coeffs, t_mid) <= 0:
            total_duration += (p_end - p_start)
            
    return total_duration

def calculate_shielding_time_robust(
    missile_pos_initial: np.ndarray,
    missile_velocity: np.ndarray,
    target_pos: np.ndarray,
    smoke_detonation_pos: np.ndarray,
    smoke_detonation_time: float,
    smoke_effective_duration: float,
    smoke_sink_velocity: np.ndarray = np.array([0., 0., -3.]),
    smoke_radius: float = 10.0
) -> float:
    """
    主函数：鲁棒地计算单个烟幕对视线的总遮蔽时长（分段解析法）。
    """
    # 1. 确定总分析区间
    T_start = smoke_detonation_time
    T_end = smoke_detonation_time + smoke_effective_duration

    # 2. 计算所有需要的多项式系数
    all_coeffs = _get_all_polynomial_coefficients(
        m0=missile_pos_initial,
        vm=missile_velocity,
        p_target=target_pos,
        p_det=smoke_detonation_pos,
        t_det=smoke_detonation_time,
        v_sink=smoke_sink_velocity,
        r_smoke=smoke_radius
    )
    N_coeffs = all_coeffs['N_coeffs']

    # 3. 寻找临界时刻 (Case 1/2 切换点)
    critical_roots = np.roots(N_coeffs)
    critical_points_in_interval = sorted([
        r.real for r in critical_roots if np.isreal(r) and T_start < r.real < T_end
    ])

    # 4. 构建求解子区间
    boundary_points = sorted(list(set([T_start, T_end] + critical_points_in_interval)))

    # 5. 分段求解并累加时长
    total_shielding_time = 0.0
    for i in range(len(boundary_points) - 1):
        sub_interval_start, sub_interval_end = boundary_points[i], boundary_points[i+1]
        
        # a. 状态判断
        t_mid = (sub_interval_start + sub_interval_end) / 2.0
        n_val_mid = np.polyval(N_coeffs, t_mid)
        
        # b. 应用对应公式
        if n_val_mid >= 0:  # Case 1
            duration = _solve_polynomial_inequality_in_interval(
                all_coeffs['P4_coeffs'],
                (sub_interval_start, sub_interval_end)
            )
        else:  # Case 2
            duration = _solve_polynomial_inequality_in_interval(
                all_coeffs['P2_A_coeffs'],
                (sub_interval_start, sub_interval_end)
            )
        total_shielding_time += duration
        
    return total_shielding_time

# =============================================================================
# 示例：使用问题一的参数进行验证
# =============================================================================
if __name__ == '__main__':
    # --- 物理常量和问题设定 ---
    G = 9.8
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    P_TRUE_TARGET = np.array([0.0, 200.0, 0.0])
    
    # --- 导弹 M1 信息 ---
    M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
    V_MISSILE_SPEED = 300.0
    direction_vec_m1 = P_FALSE_TARGET - M1_INITIAL_POS
    u_missile_m1 = direction_vec_m1 / np.linalg.norm(direction_vec_m1)
    V_VEC_MISSILE_M1 = V_MISSILE_SPEED * u_missile_m1
    
    # --- 无人机 FY1 飞行策略 (问题一) ---
    FY1_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])
    v_uav_speed = 120.0
    uav_direction = (P_FALSE_TARGET[:2] - FY1_INITIAL_POS[:2])
    uav_direction /= np.linalg.norm(uav_direction)
    v_vec_uav = np.array([uav_direction[0], uav_direction[1], 0]) * v_uav_speed
    
    # --- 烟幕弹投放与起爆信息 (问题一) ---
    t_launch = 1.5
    dt_det = 3.6
    
    p_launch = FY1_INITIAL_POS + v_vec_uav * t_launch
    v_launch = v_vec_uav
    p_detonation = p_launch + v_launch * dt_det + 0.5 * np.array([0, 0, -G]) * dt_det**2
    t_detonation = t_launch + dt_det
    
    print("--- 问题一参数 ---")
    print(f"导弹初始位置 M0: {M1_INITIAL_POS}")
    print(f"导弹速度矢量 Vm: {V_VEC_MISSILE_M1}")
    print(f"真目标位置 Pt: {P_TRUE_TARGET}")
    print(f"烟幕起爆位置 P_det: {p_detonation}")
    print(f"烟幕起爆时刻 t_det: {t_detonation:.4f} s")
    
    # --- 调用鲁棒的解析法计算遮蔽时长 ---
    robust_analytical_time = calculate_shielding_time_robust(
        missile_pos_initial=M1_INITIAL_POS,
        missile_velocity=V_VEC_MISSILE_M1,
        target_pos=P_TRUE_TARGET,
        smoke_detonation_pos=p_detonation,
        smoke_detonation_time=t_detonation,
        smoke_effective_duration=20.0,
        smoke_radius=10.0
    )
    
    print("\n--- 计算结果 ---")
    print(f"鲁棒解析法计算出的总遮蔽时长: {robust_analytical_time:.8f} s")

    # --- 对比仿真法的结果 ---
    # (这里可以放一个基于时间步进的仿真函数来验证其结果是否一致)
    # 比如，仿真结果可能是 1.8850s 左右，解析法应该给出非常接近的值。