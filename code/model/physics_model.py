# models/physics_model.py
import numpy as np
from config import G, UAV_INITIAL_POS



def calculate_shielding_time_placeholder(decision_variables, uav_id='FY1', num_bombs=1):
    """
    这是一个占位符/伪实现的核心物理模型。
    它结构完整，返回优化和可视化所需的所有数据。
    请将此函数内部的计算逻辑替换为您真实的物理模型。

    Args:
        decision_variables (np.ndarray): 决策变量向量。
        uav_id (str): 无人机编号, e.g., 'FY1'.
        num_bombs (int): 该无人机投放的干扰弹数量。

    Returns:
        tuple: (total_shielding_time, details_dict)
            - total_shielding_time (float): 总有效遮蔽时长。
            - details_dict (dict): 包含所有中间结果的字典，用于后续分析和可视化。
    """
    # --- 1. 解包决策变量 ---
    v_u = decision_variables[0]
    theta = decision_variables[1]
    t_launches = decision_variables[2 : 2 + num_bombs]
    delta_t_dets = decision_variables[2 + num_bombs : 2 + 2 * num_bombs]

    # --- 2. 模拟计算 (这里是伪代码，需要被真实模型替换) ---
    # 这是一个简单的示例目标函数，让最优解趋向于 v_u=100, theta=pi/2
    # 真实场景中，这里是你最复杂的模拟计算
    score = 0
    for t in t_launches:
        score += np.sin(t/5) # 模拟投放时间的影响
    total_shielding_time = 15 - (v_u - 105)**2 * 0.005 - (theta - np.pi/2)**2 * 2 + score

    # --- 3. 收集详细信息用于可视化 ---
    # 即使是占位符，也要生成结构正确的details_dict
    p_uav_initial = UAV_INITIAL_POS[uav_id]
    v_uav_vec = np.array([v_u * np.cos(theta), v_u * np.sin(theta), 0])
    
    launch_points = []
    detonation_points = []
    
    for i in range(num_bombs):
        t_launch = t_launches[i]
        dt_det = delta_t_dets[i]
        
        # 计算投放点
        p_launch = p_uav_initial + v_uav_vec * t_launch
        launch_points.append(p_launch.tolist())
        
        # 模拟平抛计算起爆点
        p_det_x = p_launch[0] + v_uav_vec[0] * dt_det
        p_det_y = p_launch[1] + v_uav_vec[1] * dt_det
        p_det_z = p_launch[2] - 0.5 * G * dt_det**2
        detonation_points.append([p_det_x, p_det_y, p_det_z])

    details = {
        'uav_id': uav_id,
        'uav_velocity': v_u,
        'uav_direction_rad': theta,
        'uav_direction_deg': np.rad2deg(theta),
        'launch_times': t_launches.tolist(),
        'detonation_delays': delta_t_dets.tolist(),
        'launch_points': launch_points,
        'detonation_points': detonation_points,
        'total_shielding_time': total_shielding_time,
        # 真实模型中还应包含每个烟幕的遮蔽时间段，用于绘制甘特图
        'shielding_intervals': [[t, t + 5] for t in t_launches] # 伪数据
    }
    
    return total_shielding_time, details


def cost_function3(x):
    global best_details_so_far3
    # x: [v_u, theta, t_l1, t_l2, t_l3, dt_d1, dt_d2, dt_d3]
    
    # 约束处理：投放时间间隔 >= 1s
    t_launches = x[2:5]
    if not (t_launches[0] + 1.0 <= t_launches[1] and t_launches[1] + 1.0 <= t_launches[2]):
        return 1e9  # 返回一个巨大的惩罚值

    # 调用物理模型
    shielding_time, details = calculate_shielding_time_placeholder(x, uav_id, num_bombs)
    
    # CMA-ES默认最小化，所以返回负值
    cost = -shielding_time
    
    
    # 在优化过程中记录遇到的最好结果的详情
    if not best_details_so_far3 or cost < -best_details_so_far3.get('cost', -np.inf):
            best_details_so_far3.update(details)

    return cost    