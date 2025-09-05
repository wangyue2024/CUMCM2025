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
    问题三：单无人机三弹药策略优化 (使用增强成本函数)。
    """
    print("\n--- 开始求解问题 3 (使用增强成本函数) ---")
    cfg = Config()
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)
    
    initial_guess = [100, np.pi/2, 0.0, 3.0, 6.0, 5.0, 5.0, 5.0]
    sigma0 = 5.0
    
    bounds = [
        [cfg.V_UAV_MIN, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [cfg.V_UAV_MAX, 2*np.pi, model.time_to_impact, model.time_to_impact, model.time_to_impact, 15.0, 15.0, 15.0]
    ]
    
    options = {'bounds': bounds, 'maxfevals': 5000, 'seed': 1234}
    
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 8维优化搜索已启动...")
    
    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    log_interval = 5
        # ==========================================================================
    # **核心改动**: 重新设计日志打印部分
    # ==========================================================================
    print("\n--- 优化过程追踪 ---")
    header = f"{'迭代':>5s} | {'评估数':>7s} | {'最优成本':>10s} | {'总时长':>8s} | " \
             f"{'弹1(s)':>7s} {'弹2(s)':>7s} {'弹3(s)':>7s} | " \
             f"{'弹1(m)':>7s} {'弹2(m)':>7s} {'弹3(m)':>7s}"
    print(header)
    print("-" * len(header))

    # 用于存储当前最优解的详细信息
    best_details = {
        "cost": float('inf'),
        "total_time": 0.0,
        "individual_times": [0.0, 0.0, 0.0],
        "individual_dists": [float('inf')] * 3
    }

    while not es.stop():
        solutions = es.ask()
        
        # 计算成本和遮蔽时长
        results = [model.cost_function_q3_enhanced(s) for s in solutions]
        costs = [r[0] for r in results]
        
        es.tell(solutions, costs)
        
        # 检查是否需要更新和打印日志
        if es.countiter % log_interval == 0:
            # 获取当前优化器认为的最优解
            current_best_solution = es.result.xbest
            
            # 为了获取详细的分解信息，我们需要对当前最优解重新计算一次
            # 这会带来微小的额外计算开销，但对于调试是值得的
            # 注意：这里我们不需要高精度，使用默认的优化步长即可
            v, th, t1, t2, t3, dt1, dt2, dt3 = current_best_solution
            
            # 检查约束，如果最优解违反约束，则不进行详细计算
            if not (t2 >= t1 + cfg.MIN_LAUNCH_INTERVAL and t3 >= t2 + cfg.MIN_LAUNCH_INTERVAL):
                 print(f"{es.countiter:5d} | {es.countevals:7d} | {es.result.fbest:10.4f} | {'N/A':>8s} | (违反约束)")
                 continue

            total_time, individual_dists, _ = model.calculate_shielding_metrics_p3(
                v, th, [t1, t2, t3], [dt1, dt2, dt3]
            )
            
            # 计算每个烟幕弹的独立遮蔽时间 (这是一个近似计算，但用于调试足够了)
            # 我们通过单独模拟每个烟幕弹来实现
            time1, _, _ = model.calculate_shielding_metrics(v, th, [t1], [dt1])
            time2, _, _ = model.calculate_shielding_metrics(v, th, [t2], [dt2])
            time3, _, _ = model.calculate_shielding_metrics(v, th, [t3], [dt3])
            individual_times = [time1, time2, time3]

            # 打印详细日志
            log_line = f"{es.countiter:5d} | {es.countevals:7d} | {es.result.fbest:10.4f} | {total_time:8.2f}s | " \
                       f"{individual_times[0]:7.2f} {individual_times[1]:7.2f} {individual_times[2]:7.2f} | " \
                       f"{individual_dists[0]:7.1f} {individual_dists[1]:7.1f} {individual_dists[2]:7.1f}"
            print(log_line)


    best_solution = es.result.xbest
    
    # 高精度验证
    final_shielding_time, _, final_details = model.calculate_shielding_metrics_p3(
        best_solution[0], best_solution[1],
        best_solution[2:5], best_solution[5:8],
        time_step=cfg.SIM_TS_ACCURACY
    )
    
    print("-" * 70)
    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"  弹2: 投放时间={t2:.2f}s, 延迟={dt2:.2f}s")
    print(f"  弹3: 投放时间={t3:.2f}s, 延迟={dt3:.2f}s")
    print(f"最终精确验证的最大遮蔽时长: {final_shielding_time:.4f} s")

    results = {
        "problem_id": 3,
        "best_solution_vector": best_solution,
        "max_shielding_time": final_shielding_time,
        "details": final_details,
        "log": es
    }
    return results

def solve_problem_4():
    print("\n--- 问题 4 求解器 (待实现) ---")
    return {"problem_id": 4}

def solve_problem_5():
    print("\n--- 问题 5 求解器 (待实现) ---")
    return {"problem_id": 5}