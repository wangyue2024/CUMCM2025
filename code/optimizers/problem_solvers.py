# optimizers/problem_solvers.py
import numpy as np
import cma
import pandas as pd
from models.physics_model import cost_function3,cost_function4
from config import UAV_VELOCITY_RANGE

# 全局变量用于在优化过程中存储最佳详情
best_details_so_far = {}

def solve_problem_3():
    """
    求解问题3: FY1投放3枚干扰弹，实施对M1的干扰。
    """
    print("--- Starting Optimization for Problem 3 ---")
    uav_id = 'FY1'
    num_bombs = 3
    
    # --- 1. 定义目标函数包装器 ---
    def objective_function(x):
        global best_details_so_far
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
        if not best_details_so_far or cost < -best_details_so_far.get('total_shielding_time', -np.inf):
             best_details_so_far.update(details)

        return cost

    # --- 2. 设置优化器参数 ---
    # 初始猜测解 (8个变量)
    initial_guess = [
        np.mean(UAV_VELOCITY_RANGE), # v_u
        np.pi / 4,                   # theta
        5.0, 7.0, 9.0,               # t_launches
        3.0, 3.0, 3.0                # delta_t_dets
    ]
    sigma0 = 3.0 # 初始搜索步长
    
    # 边界条件
    lower_bounds = [UAV_VELOCITY_RANGE[0], 0, 0, 1, 2, 1, 1, 1]
    upper_bounds = [UAV_VELOCITY_RANGE[1], 2 * np.pi, 40, 41, 42, 10, 10, 10]
    
    options = {'bounds': [lower_bounds, upper_bounds], 'maxfevals': 2000, 'verbose': -9}

    # --- 3. 运行优化器 ---
    best_solution_x, es = cma.fmin2(cost_function3, initial_guess, sigma0, options)
    
    # --- 4. 结果处理与输出 ---
    print("Optimization finished.")
    final_details = best_details_so_far # 获取优化过程中记录的最佳结果详情
    
    print(f"Max shielding time found: {final_details['total_shielding_time']:.4f} s")
    
    # 格式化输出到 result1.xlsx
    output_data = []
    for i in range(num_bombs):
        row = {
            '无人机编号': final_details['uav_id'],
            '无人机运动方向(度)': f"{final_details['uav_direction_deg']:.2f}",
            '无人机运动速度(m/s)': f"{final_details['uav_velocity']:.2f}",
            '烟幕干扰弹投放点的x坐标(m)': f"{final_details['launch_points'][i][0]:.2f}",
            '烟幕干扰弹投放点的y坐标(m)': f"{final_details['launch_points'][i][1]:.2f}",
            '烟幕干扰弹投放点的z坐标(m)': f"{final_details['launch_points'][i][2]:.2f}",
            '烟幕干扰弹起爆点的x坐标(m)': f"{final_details['detonation_points'][i][0]:.2f}",
            '烟幕干扰弹起爆点的y坐标(m)': f"{final_details['detonation_points'][i][1]:.2f}",
            '烟幕干扰弹起爆点的z坐标(m)': f"{final_details['detonation_points'][i][2]:.2f}",
            '有效干扰时长(s)': f"{final_details['total_shielding_time']:.4f}" if i == 0 else "" # 只在第一行显示总时长
        }
        output_data.append(row)
        
    df = pd.DataFrame(output_data)
    df.to_excel('results/result1.xlsx', index=False)
    print("Results saved to results/result1.xlsx")
    
    # 返回最终结果，方便main.py调用可视化
    return final_details, es

# 你可以在这里添加 solve_problem_4() 和 solve_problem_5()
# 它们将遵循与 solve_problem_3() 类似的结构，只是决策变量的定义和数量会变化。