# optimizers/problem_solvers.py

import numpy as np
import cma
from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical
# from utils import visualization # (如果需要保存结果)

def solve_problem_1():
    """问题一：固定策略的精确计算。"""
    print("\n--- 开始求解问题 1 (解析法精确计算) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    
    # 问题一的固定参数
    uav_speed = 120.0
    fy1_pos = cfg.UAV_INITIAL_POS['FY1']
    target_pos = cfg.P_FALSE_TARGET
    direction = target_pos[:2] - fy1_pos[:2]
    uav_theta = np.arctan2(direction[1], direction[0])
    launch_time = 1.5
    det_delay = 3.6

    # 使用模型的 get_final_details 方法来计算所有物理量
    solution_vector = [uav_speed, uav_theta, launch_time, det_delay]
    details = model.get_final_details('FY1', solution_vector)
    
    shielding_time = details["total_shielding_time"]
    
    print("计算完成！")
    print(f"烟幕弹对 M1 的有效遮蔽时长: {shielding_time:.8f} s")
    
    return {"problem_id": 1, "max_shielding_time": shielding_time, "details": details}

def solve_problem_2():
    """问题二：单无人机单弹药策略优化。"""
    print("\n--- 开始求解问题 2 (解析法 + CMA-ES 优化) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    
    initial_guess = [105, np.pi / 4, 5.0, 5.0]
    sigma0 = 3.0
    bounds = [
        [cfg.V_UAV_MIN, 0, 0.1, 0.1],
        [cfg.V_UAV_MAX, 2 * np.pi, model.time_to_impact - 5, 20.0]
    ]
    
    # 性能飞跃，我们可以承担得起更多的评估次数和更大的种群
    options = {'bounds': bounds, 'maxfevals': 10000, 'popsize': 30, 'seed': 42, 'verbose': -9}
    
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 4维优化搜索已启动...")
    
    # (可以加入迭代打印日志)
    es.optimize(model.cost_function_q2)
    
    best_solution = es.result.xbest
    final_details = model.get_final_details('FY1', best_solution)
    final_shielding_time = final_details["total_shielding_time"]

    print("优化完成！")
    v, th, t1, dt1 = best_solution
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"最终精确验证的最大遮蔽时长: {final_shielding_time:.4f} s")

    return {"problem_id": 2, "max_shielding_time": final_shielding_time, "details": final_details, "log": es}

def solve_problem_3():
    """问题三：单无人机三弹药策略优化。"""
    print("\n--- 开始求解问题 3 (解析法 + CMA-ES 优化) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    
    initial_guess = [105, np.pi/4, 5.0, 10.0, 15.0, 5.0, 5.0, 5.0]
    sigma0 = 5.0
    bounds_lower = [cfg.V_UAV_MIN, 0, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1]
    bounds_upper = [cfg.V_UAV_MAX, 2*np.pi, model.time_to_impact-10, model.time_to_impact-9, model.time_to_impact-8, 20.0, 20.0, 20.0]
    
    # 更高维度，需要更多评估
    options = {'bounds': [bounds_lower, bounds_upper], 'maxfevals': 30000, 'popsize': 50, 'seed': 42, 'verbose': -9}
    
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 8维优化搜索已启动...")
    
    es.optimize(model.cost_function_q3)
    
    best_solution = es.result.xbest
    final_details = model.get_final_details('FY1', best_solution)
    final_shielding_time = final_details["total_shielding_time"]

    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"  弹2: 投放时间={t2:.2f}s, 延迟={dt2:.2f}s")
    print(f"  弹3: 投放时间={t3:.2f}s, 延迟={dt3:.2f}s")
    print(f"最终精确验证的最大遮蔽时长: {final_shielding_time:.4f} s")

    return {"problem_id": 3, "max_shielding_time": final_shielding_time, "details": final_details, "log": es}