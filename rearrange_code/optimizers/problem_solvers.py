# optimizers/problem_solvers.py

import cma
import numpy as np
from models.physics_model import PhysicsModelWithCylinder
from config import Config

def solve_problem_1():
    """
    问题一：确定性仿真计算。
    根据题目给定的固定策略，计算有效遮蔽时长。
    """
    print("--- 开始求解问题 1 (高精度仿真) ---")
    cfg = Config()
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)

    # 题目给定的策略
    uav_speed = 120.0
    direction_vec = cfg.P_FALSE_TARGET - cfg.UAV_INITIAL_POS['FY1']
    uav_theta = np.arctan2(direction_vec[1], direction_vec[0])
    launch_time = 1.5
    det_delay = 3.6

    # 使用高精度时间步长进行计算
    shielding_time, _, details = model.calculate_shielding_metrics(
        uav_speed, uav_theta, [launch_time], [det_delay],
        time_step=cfg.SIM_TS_ACCURACY
    )

    print(f"策略: 速度={uav_speed} m/s, 方向={np.rad2deg(uav_theta):.2f}°, 投放时间={launch_time}s, 延迟={det_delay}s")
    print(f"计算结果: 有效遮蔽时长 = {shielding_time:.4f} s")
    
    results = {
        "problem_id": 1,
        "shielding_time": shielding_time,
        "details": details
    }
    return results

def solve_problem_2():
    """
    问题二：单无人机单弹药策略优化。
    使用CMA-ES算法寻找最优策略，并适度输出迭代过程信息。
    """
    print("\n--- 开始求解问题 2 (CMA-ES 优化) ---")
    cfg = Config()
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)
    
    # 决策变量: [速度, 方向角, 投放时间, 起爆延迟]
    initial_guess = [105, np.pi/4, 3.0, 5.0]
    sigma0 = 5.0
    
    bounds = [
        [cfg.V_UAV_MIN, 0, 0, 0.0],
        [cfg.V_UAV_MAX, 2*np.pi, model.time_to_impact , 20.0]
    ]
    
    # 注意：我们将 'verbose': -9 从options中移除，因为我们要自定义输出
    options = {'bounds': bounds, 'maxfevals': 2000, 'seed': 1234}
    
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 优化搜索已启动...")
    
    # ==========================================================================
    # **核心改动**: 使用 ask-tell 循环代替 fmin2 以便自定义输出
    # ==========================================================================
    
    # 1. 初始化CMA-ES策略对象
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)

    # 2. 设置日志打印间隔和表头
    log_interval = 10  # 每20代 (generation) 打印一次信息
    print("\n--- 优化过程追踪 ---")
    print(f"{'迭代':>3s} | {'评估次数':>4s} | {'当前最优成本':>6s} | {'对应遮蔽时长':>8s} | {'步长(Sigma)':>10s}")
    print("-" * 70)

    # 3. 手动执行优化循环
    while not es.stop():
        # 从当前策略分布中获取新一代的解向量
        solutions = es.ask()
        
        # 计算每个解的成本值（调用物理模型）
        costs = [model.cost_function_q2(s) for s in solutions]
        
        # 将解和对应的成本值反馈给优化器，以更新策略分布
        es.tell(solutions, costs)
        
        # 按设定的间隔打印调试信息
        if es.countiter % log_interval == 0:
            # 成本值是负的遮蔽时长，我们把它转为正的，方便观察
            # 如果成本值为正（来自奖励塑造），说明遮蔽时间为0
            current_best_time = -es.result.fbest if es.result.fbest < 0 else 0.0
            print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | {current_best_time:13.4f}s | {es.sigma:12.6f}")

    # 4. 优化结束后，获取最终结果
    best_solution = es.result.xbest
    # fmin2会返回-es.result.fbest，我们手动保持一致
    max_shielding_time = -es.result.fbest if es.result.fbest < 0 else 0.0
    
    # ==========================================================================
    # 循环结束，后续处理不变
    # ==========================================================================

    # 使用最优解进行一次高精度仿真以获取最终的精确时间和细节
    final_shielding_time, _, final_details = model.calculate_shielding_metrics(
        best_solution[0], best_solution[1], [best_solution[2]], [best_solution[3]],
        time_step=cfg.SIM_TS_ACCURACY
    )
    
    print("-" * 70)
    print("优化完成！")
    print(f"最优策略: 速度={best_solution[0]:.2f} m/s, 方向={np.rad2deg(best_solution[1]):.2f}°, 投放时间={best_solution[2]:.2f}s, 延迟={best_solution[3]:.2f}s")
    print(f"优化过程找到的最大遮蔽时长 (基于优化步长): {max_shielding_time:.4f} s")
    print(f"最终精确验证的最大遮蔽时长 (基于高精度步长): {final_shielding_time:.4f} s")

    results = {
        "problem_id": 2,
        "best_solution_vector": best_solution,
        "max_shielding_time": final_shielding_time, # 返回精确值
        "details": final_details,
        "log": es  # 保存优化过程对象，用于绘制收敛曲线
    }
    return results

# --- 为后续问题预留的占位函数 ---

def solve_problem_3():
    """
    问题三：单无人机多弹药策略优化。
    使用增强版成本函数和CMA-ES算法寻找最优策略，确保所有烟幕弹都参与优化。
    """
    print("\n--- 开始求解问题 3 (增强型 CMA-ES 优化) ---")
    cfg = Config()
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)
    
    # 决策变量: [速度, 方向角, 投放时间1, 起爆延迟1, 投放时间2, 起爆延迟2, ...]
    # 这里我们设置为3枚烟幕弹
    num_smokes = 3
    
    # 初始猜测值
    initial_guess = [105, np.pi/4]  # 速度和方向角
    for i in range(num_smokes):
        # 均匀分布的投放时间和相同的起爆延迟
        launch_time = 2.0 + i * 3.0  # 从2秒开始，每3秒投放一枚
        det_delay = 5.0
        initial_guess.extend([launch_time, det_delay])
    
    sigma0 = 5.0
    
    # 设置边界
    lower_bounds = [cfg.V_UAV_MIN, 0]  # 速度和方向角的下界
    upper_bounds = [cfg.V_UAV_MAX, 2*np.pi]  # 速度和方向角的上界
    
    for i in range(num_smokes):
        # 投放时间的下界为0，上界为导弹预计撞击时间
        # 起爆延迟的下界为0，上界为20秒
        lower_bounds.extend([0, 0.0])
        upper_bounds.extend([model.time_to_impact, 20.0])
    
    bounds = [lower_bounds, upper_bounds]
    
    # 优化器选项
    options = {
        'bounds': bounds, 
        'maxfevals': 3000, 
        'seed': 1234,
        'popsize': 20,     # 增加种群大小以提高探索能力
        'tolx': 1e-4,      # 参数变化容差
        'tolfun': 1e-4     # 函数值变化容差
    }
    
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 优化搜索已启动...")
    print(f"优化{num_smokes}枚烟幕弹的投放策略")
    
    # 初始化CMA-ES策略对象
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)

    # 设置日志打印间隔和表头
    log_interval = 10  # 每10代打印一次信息
    print("\n--- 优化过程追踪 ---")
    print(f"{'迭代':>3s} | {'评估次数':>4s} | {'当前最优成本':>6s} | {'对应遮蔽时长':>8s} | {'步长(Sigma)':>10s}")
    print("-" * 70)

    # 手动执行优化循环
    while not es.stop():
        # 从当前策略分布中获取新一代的解向量
        solutions = es.ask()
        
        # 计算每个解的成本值（调用增强版物理模型）
        costs = [model.cost_function_q3_enhanced(s) for s in solutions]
        
        # 将解和对应的成本值反馈给优化器，以更新策略分布
        es.tell(solutions, costs)
        
        # 按设定的间隔打印调试信息
        if es.countiter % log_interval == 0:
            # 成本值是负的遮蔽时长，我们把它转为正的，方便观察
            # 如果成本值为正（来自奖励塑造），说明遮蔽时间为0
            current_best_time = -es.result.fbest if es.result.fbest < 0 else 0.0
            print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | {current_best_time:13.4f}s | {es.sigma:12.6f}")

    # 优化结束后，获取最终结果
    best_solution = es.result.xbest
    max_shielding_time = -es.result.fbest if es.result.fbest < 0 else 0.0
    
    # 使用最优解进行一次高精度仿真以获取最终的精确时间和细节
    launch_times = [best_solution[2 + i*2] for i in range(num_smokes)]
    det_delays = [best_solution[3 + i*2] for i in range(num_smokes)]
    
    # 使用增强版成本函数进行高精度验证
    final_shielding_time, _, final_details = model.calculate_shielding_metrics(
        best_solution[0], best_solution[1], launch_times, det_delays,
        time_step=cfg.SIM_TS_ACCURACY
    )
    
    # 获取详细的遮蔽指标，用于分析每个烟幕弹的贡献
    _, min_miss_distances = model.calculate_shielding_metrics_detailed(
        best_solution, time_step=cfg.SIM_TS_ACCURACY
    )
    
    print("-" * 70)
    print("优化完成！")
    print(f"最优策略: 速度={best_solution[0]:.2f} m/s, 方向={np.rad2deg(best_solution[1]):.2f}°")
    for i in range(num_smokes):
        print(f"烟幕弹 {i+1}: 投放时间={launch_times[i]:.2f}s, 延迟={det_delays[i]:.2f}s")
    print(f"优化过程找到的最大遮蔽时长 (基于优化步长): {max_shielding_time:.4f} s")
    print(f"最终精确验证的最大遮蔽时长 (基于高精度步长): {final_shielding_time:.4f} s")
    
    # 分析每个烟幕弹的贡献
    print("\n各烟幕弹贡献分析:")
    for i, dist in enumerate(min_miss_distances):
        if dist <= model.config.smoke_radius:
            print(f"烟幕弹 {i+1} 有效贡献遮蔽效果，最小未命中距离: {dist:.2f}m")
        else:
            print(f"烟幕弹 {i+1} 未有效贡献遮蔽效果，最小未命中距离: {dist:.2f}m > {model.config.smoke_radius:.2f}m")

    results = {
        "problem_id": 3,
        "best_solution_vector": best_solution,
        "max_shielding_time": final_shielding_time, # 返回精确值
        "details": final_details,
        "min_miss_distances": min_miss_distances,
        "smoke_contributions": [dist <= model.config.smoke_radius for dist in min_miss_distances],
        "log": es  # 保存优化过程对象，用于绘制收敛曲线
    }
    return results

def solve_problem_4():
    print("\n--- 问题 4 求解器 (待实现) ---")
    return {"problem_id": 4}

def solve_problem_5():
    print("\n--- 问题 5 求解器 (待实现) ---")
    return {"problem_id": 5}