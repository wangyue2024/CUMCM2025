# problem_solvers.py

import numpy as np
import cma
from config import Config
from analytical_model import PhysicsModelAnalytical

def solve_problem_1():
    """问题一：固定策略下的遮蔽时长计算（使用解析法）。"""
    print("\n--- 开始求解问题 1 (解析法精确计算) ---")
    cfg = Config()
    
    # 1. 定义问题一的固定参数
    v_uav_speed = 120.0
    t_launch = 1.5
    dt_det = 3.6
    
    # 2. 计算无人机和烟幕的运动学状态
    uav_pos_0 = cfg.UAV_INITIAL_POS['FY1']
    uav_dir = (cfg.P_FALSE_TARGET[:2] - uav_pos_0[:2])
    uav_dir /= np.linalg.norm(uav_dir)
    v_vec_uav = np.array([uav_dir[0], uav_dir[1], 0]) * v_uav_speed
    
    p_launch = uav_pos_0 + v_vec_uav * t_launch
    p_det = p_launch + v_vec_uav * dt_det + 0.5 * np.array([0, 0, -cfg.G]) * dt_det**2
    t_det = t_launch + dt_det

    # 3. 使用解析模型直接计算
    model = PhysicsModelAnalytical(missile_id='M1', uav_id='FY1')
    intervals = model._calculate_single_smoke_intervals_robust(p_det, t_det)
    total_shielding_time = sum(end - start for start, end in intervals)
    
    print("计算完成！")
    print(f"烟幕起爆时刻: {t_det:.4f} s")
    print(f"烟幕起爆位置: {np.round(p_det, 2)}")
    print(f"有效遮蔽时间区间: {[(round(s, 4), round(e, 4)) for s, e in intervals]}")
    print(f"总有效遮蔽时长: {total_shielding_time:.8f} s")
    return {"total_shielding_time": total_shielding_time}

def solve_problem_2():
    """问题二：单无人机单弹药策略优化。"""
    print("\n--- 开始求解问题 2 (CMA-ES + 解析模型) ---")
    model = PhysicsModelAnalytical(missile_id='M1', uav_id='FY1')
    
    # 决策变量: [速度, 方向(rad), 投放时间, 延迟时间]
    bounds = [
        [model.config.V_UAV_MIN, 0, 0.1, 0.1],
        [model.config.V_UAV_MAX, 2 * np.pi, model.time_to_impact - 25, 20.0]
    ]
    initial_guess = [100, np.pi / 4, 5.0, 5.0]
    sigma0 = 3.0
    # 利用解析解的速度优势，增加迭代次数和种群规模
    options = {'bounds': bounds, 'maxfevals': 3000, 'popsize': 22, 'seed': 42, 'verbose': -9}
    
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 4维优化搜索已启动...")
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    es.optimize(model.cost_function_q2)
    
    best_solution = es.result.xbest
    final_cost = es.result.fbest
    
    print("优化完成！")
    v, th, t_l, dt_d = best_solution
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°, 投放时间={t_l:.2f}s, 延迟={dt_d:.2f}s")
    print(f"优化找到的最大遮蔽时长 (可能包含奖励塑造影响): {-final_cost:.4f} s")
    
    # 精确验证最终结果
    final_time = -model.cost_function_q2(best_solution)
    print(f"最终精确验证的最大遮蔽时长: {final_time:.4f} s")
    return {"max_shielding_time": final_time, "best_solution": best_solution}

def solve_problem_3():
    """问题三：单无人机三弹药策略优化。"""
    print("\n--- 开始求解问题 3 (CMA-ES + 解析模型 + 协同奖励) ---")
    model = PhysicsModelAnalytical(missile_id='M1', uav_id='FY1')

    # 决策变量: [v, th, t1, t2, t3, dt1, dt2, dt3]
    bounds_lower = [model.config.V_UAV_MIN, 0, 0.1, 1.1, 2.1, 0.1, 0.1, 0.1]
    bounds_upper = [model.config.V_UAV_MAX, 2*np.pi, 40, 45, 50, 20, 20, 20]
    bounds = [bounds_lower, bounds_upper]
    
    initial_guess = [100, np.pi/4, 5, 15, 25, 5, 5, 5]
    sigma0 = 5.0
    # 更大的搜索空间需要更多的评估次数和更大的种群
    options = {'bounds': bounds, 'maxfevals': 15000, 'popsize': 40, 'seed': 42, 'verbose': -9}

    print(f"8维优化搜索已启动，这将需要一些时间...")
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    es.optimize(model.cost_function_q3)
    
    best_solution = es.result.xbest
    final_cost = es.result.fbest

    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  烟幕弹 1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"  烟幕弹 2: 投放时间={t2:.2f}s, 延迟={dt2:.2f}s")
    print(f"  烟幕弹 3: 投放时间={t3:.2f}s, 延迟={dt3:.2f}s")
    print(f"优化找到的最大遮蔽时长 (可能包含奖励塑造影响): {-final_cost:.4f} s")

    # 精确验证最终结果
    final_time = -model.cost_function_q3(best_solution)
    print(f"最终精确验证的最大遮蔽时长: {final_time:.4f} s")
    return {"max_shielding_time": final_time, "best_solution": best_solution}