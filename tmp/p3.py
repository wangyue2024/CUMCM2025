import numpy as np
import cma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# ==============================================================================
# 1. 配置与常量 (CONFIG)
# ==============================================================================
# --- 目标与威胁 ---
P_TRUE = np.array([0, 200, 0])
P_M1_0 = np.array([20000, 0, 2000])
P_TARGET = np.array([0, 0, 0])
V_M = 300.0

# --- 无人机与干扰弹 ---
UAV_INITIAL_POS = np.array([17800, 0, 1800])
V_UAV_MIN, V_UAV_MAX = 70, 140
LAUNCH_INTERVAL_MIN = 1.0
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
GRAVITY = 9.8

# --- 模拟与优化参数 ---
SIMULATION_DT = 0.1  # 时间步长 (s)，越小越准但越慢
OPTIMIZER_MAX_EVALS = 3000 # 增加评估次数以获得更好的探索

# --- 预计算常量 ---
M1_TRAVEL_VECTOR = P_TARGET - P_M1_0
M1_TRAVEL_DISTANCE = np.linalg.norm(M1_TRAVEL_VECTOR)
U_M1 = M1_TRAVEL_VECTOR / M1_TRAVEL_DISTANCE
T_END_SIMULATION = M1_TRAVEL_DISTANCE / V_M

# ==============================================================================
# 2. 物理模型与COST FUNCTION (改进版)
# ==============================================================================
def point_to_segment_distance(point, seg_start, seg_end):
    """计算点到线段的最短距离"""
    seg_vec = seg_end - seg_start
    point_vec = point - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq < 1e-9: return np.linalg.norm(point_vec)
    k = np.dot(point_vec, seg_vec) / seg_len_sq
    if k < 0: return np.linalg.norm(point_vec)
    if k > 1: return np.linalg.norm(point - seg_end)
    return np.linalg.norm(point - (seg_start + k * seg_vec))

def physics_based_cost_function(x):
    """
    基于完整物理模型的cost function。
    现在会额外返回整个过程中的最小接近距离。
    """
    # 1. 解包决策变量
    v_u, theta = x[0], x[1]
    t_launches = x[2:5]
    delta_t_dets = x[5:8]

    # 2. 计算所有烟幕的关键参数
    uav_vel_vec = np.array([v_u * np.cos(theta), v_u * np.sin(theta), 0])
    t_dets = t_launches + delta_t_dets
    p_launches = UAV_INITIAL_POS + uav_vel_vec * t_launches[:, np.newaxis]
    p_dets_x = p_launches[:, 0] + uav_vel_vec[0] * delta_t_dets
    p_dets_y = p_launches[:, 1] + uav_vel_vec[1] * delta_t_dets
    p_dets_z = p_launches[:, 2] - 0.5 * GRAVITY * delta_t_dets**2
    p_dets = np.vstack([p_dets_x, p_dets_y, p_dets_z]).T

    # 3. 时间离散化模拟
    time_steps = np.arange(0, T_END_SIMULATION, SIMULATION_DT)
    total_shielding_time = 0.0
    shielding_intervals = [[] for _ in range(3)]
    min_distance_overall = float('inf')

    for t in time_steps:
        p_m1_t = P_M1_0 + U_M1 * V_M * t
        is_shielded_this_step = False
        min_dist_this_step = float('inf')
        
        for i in range(3):
            if t_dets[i] <= t <= t_dets[i] + T_SMOKE_EFFECTIVE:
                c_s_t = p_dets[i].copy()
                c_s_t[2] -= V_SMOKE_SINK * (t - t_dets[i])
                dist = point_to_segment_distance(c_s_t, p_m1_t, P_TRUE)
                min_dist_this_step = min(min_dist_this_step, dist)
                if dist <= R_SMOKE:
                    is_shielded_this_step = True
                    shielding_intervals[i].append(t)
        
        min_distance_overall = min(min_distance_overall, min_dist_this_step)
        if is_shielded_this_step:
            total_shielding_time += SIMULATION_DT

    # 4. 准备返回的详细信息
    details = {
        "uav_velocity": v_u, "uav_direction_rad": theta,
        "launch_times": t_launches, "detonation_delays": delta_t_dets,
        "effective_shielding_time": total_shielding_time,
        "launch_points": p_launches, "detonation_points": p_dets,
        "uav_vel_vec": uav_vel_vec, "shielding_intervals": shielding_intervals,
        "min_distance_overall": min_distance_overall
    }
    return total_shielding_time, details

# ==============================================================================
# 3. 优化器接口 (改进版)
# ==============================================================================
eval_count = 0
def objective_function_for_cma(x):
    """
    包装器函数，采用分段目标函数引导优化。
    """
    global eval_count
    eval_count += 1
    
    t1, t2, t3 = x[2:5]
    if not (t2 - t1 >= LAUNCH_INTERVAL_MIN and t3 - t2 >= LAUNCH_INTERVAL_MIN and t1 < t2 < t3):
        return 1000.0
    
    total_time, details = physics_based_cost_function(x)
    
    if total_time > 0:
        cost = -total_time
    else:
        min_dist = details["min_distance_overall"]
        cost = min_dist - R_SMOKE
    
    if eval_count % 50 == 0:
        best_cost = es.result.fbest if hasattr(es, 'result') else float('inf')
        if best_cost < 0:
            print(f"  Eval #{eval_count}: Best time = {-best_cost:.2f}s", end="")
        else:
            print(f"  Eval #{eval_count}: Best approach = {best_cost + R_SMOKE:.2f}m", end="")
        
        if cost < 0:
            print(f", Current trial time = {total_time:.2f}s")
        else:
            print(f", Current trial min_dist = {min_dist:.2f}m")
            
    return cost

# ==============================================================================
# 4. 可视化函数 (与之前版本相同)
# ==============================================================================
def visualize_results(details, es):
    """统一的可视化入口"""
    print("\n--- Generating Visualizations ---")
    
    # 1. 3D战术图
    print("  - Plotting 3D Tactical View...")
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    m1_path_t = np.linspace(0, T_END_SIMULATION, 200)
    m1_path = P_M1_0 + m1_path_t[:, np.newaxis] * U_M1 * V_M
    ax1.plot(m1_path[:, 0], m1_path[:, 1], m1_path[:, 2], 'r-', label='Missile M1 Path')
    ax1.scatter(P_M1_0[0], P_M1_0[1], P_M1_0[2], c='red', marker='x', s=100, label='M1 Start')
    ax1.scatter(P_TARGET[0], P_TARGET[1], P_TARGET[2], c='black', marker='s', s=100, label='False Target')
    ax1.scatter(P_TRUE[0], P_TRUE[1], P_TRUE[2], c='cyan', marker='P', s=150, label='True Target')
    max_time = max(details['launch_times']) + 1
    uav_path_t = np.linspace(0, max_time, 100)
    uav_path = UAV_INITIAL_POS + uav_path_t[:, np.newaxis] * details['uav_vel_vec']
    ax1.plot(uav_path[:, 0], uav_path[:, 1], uav_path[:, 2], 'g--', label='UAV Path')
    ax1.scatter(UAV_INITIAL_POS[0], UAV_INITIAL_POS[1], UAV_INITIAL_POS[2], c='green', s=100, label='UAV Start')
    l_points = details['launch_points']
    d_points = details['detonation_points']
    ax1.scatter(l_points[:, 0], l_points[:, 1], l_points[:, 2], c='blue', s=80, label='Launch Points', depthshade=False)
    ax1.scatter(d_points[:, 0], d_points[:, 1], d_points[:, 2], c='purple', s=150, marker='*', label='Detonation Points', depthshade=False)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Tactical Visualization')

    # 2. 遮蔽时间甘特图
    print("  - Plotting Shielding Gantt Chart...")
    ax2 = fig.add_subplot(1, 2, 2)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, intervals in enumerate(details['shielding_intervals']):
        if not intervals: continue
        start_time = intervals[0]
        for j in range(1, len(intervals)):
            if intervals[j] - intervals[j-1] > SIMULATION_DT * 1.5:
                ax2.barh(f'Grenade {i+1}', width=intervals[j-1] - start_time, left=start_time, color=colors[i], alpha=0.6)
                start_time = intervals[j]
        ax2.barh(f'Grenade {i+1}', width=intervals[-1] - start_time, left=start_time, color=colors[i], alpha=0.6)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Smoke Grenade')
    ax2.set_title(f'Shielding Gantt Chart (Total: {details["effective_shielding_time"]:.2f}s)')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # 3. 收敛曲线
    print("  - Plotting CMA-ES Convergence...")
    # 清理旧的日志文件以避免绘图警告
    if os.path.exists('outcmaes'):
        import shutil
        shutil.rmtree('outcmaes')
    es.logger.add() # 确保在绘图前记录了最后的数据
    es.plot()

# ==============================================================================
# 5. 主求解器 (改进版)
# ==============================================================================
# ==============================================================================
# 5. 主求解器 (最终修正版 v3 - 强制日志记录)
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Optimization for Problem 3 ---")
    print(f"Simulation Time Step (dt): {SIMULATION_DT}s")
    print(f"Optimizer Max Evaluations: {OPTIMIZER_MAX_EVALS}")
    start_time = time.time()

    # 清理旧的日志文件以避免绘图警告和错误
    log_dir = 'outcmaes'
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)

    initial_guess = [105, 1.6, 15, 25, 35, 6, 6, 6]
    sigma0 = 8.0
    bounds = [[V_UAV_MIN, 0, 0, LAUNCH_INTERVAL_MIN, 2*LAUNCH_INTERVAL_MIN, 1, 1, 1],
              [V_UAV_MAX, 2*np.pi, 50, 50, 50, 10, 10, 10]]
    options = {'bounds': bounds, 'maxfevals': OPTIMIZER_MAX_EVALS, 'verbose': -9}

    global es
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    
    # --- 显式创建和配置日志记录器 ---
    # 这会告诉优化器将数据记录到 'outcmaes' 文件夹
    logger = cma.CMADataLogger().register(es)

    # --- 使用手动的 ask-tell 循环 ---
    # 并在每一步手动更新日志
    while not es.stop():
        try:
            solutions = es.ask()
            fitnesses = [objective_function_for_cma(s) for s in solutions]
            es.tell(solutions, fitnesses)
            
            # --- 手动更新日志 ---
            logger.add() # 这一行是关键！

        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            break

    end_time = time.time()
    print(f"\nOptimization finished in {(end_time - start_time)/60:.2f} minutes.")

    # 在结束时最后刷新一次日志，确保所有数据写入文件

    # --- 结果分析与展示 ---
    best_solution_x = es.result.xbest
    final_time, final_details = physics_based_cost_function(best_solution_x)
    
    print("\n" + "="*20 + " Optimal Strategy Found " + "="*20)
    print(f"Maximum Effective Shielding Time: {final_details['effective_shielding_time']:.4f} s")
    print(f"Closest Approach Distance: {final_details['min_distance_overall']:.2f} m")
    print("\n--- Strategy Details ---")
    print(f"UAV Velocity: {final_details['uav_velocity']:.2f} m/s")
    print(f"UAV Direction: {np.rad2deg(final_details['uav_direction_rad']):.2f} degrees")
    for i in range(3):
        print(f"  - Grenade {i+1}: Launch at {final_details['launch_times'][i]:.2f}s, Detonation delay {final_details['detonation_delays'][i]:.2f}s")

    # 调用可视化
    # 此时logger应该已经包含了完整的数据
    visualize_results(final_details, es)