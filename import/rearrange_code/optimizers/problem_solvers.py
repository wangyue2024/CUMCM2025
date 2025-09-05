# optimizers/problem_solvers.py

import numpy as np
import cma
import time
from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical

class OptimizationWrapper:
    """
    一个封装了变量归一化和约束处理的优化接口。
    优化器将在这个类提供的无约束、归一化的空间中进行操作。
    """
    def __init__(self, model: PhysicsModelAnalytical, problem_id: int):
        self.model = model
        self.cfg = model.config
        self.problem_id = problem_id
        self.bounds_norm = [[0.0] * self._get_dim(), [1.0] * self._get_dim()]

    def _get_dim(self):
        return 4 if self.problem_id == 2 else 8

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        """将归一化的向量解码为真实的物理参数向量。"""
        x_phys = np.zeros_like(x_norm)
        
        # 解码速度和角度
        x_phys[0] = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
        x_phys[1] = x_norm[1] * 2 * np.pi

        if self.problem_id == 2:
            # 解码 t_launch, dt_det
            x_phys[2] = x_norm[2] * (self.model.time_to_impact - 5)
            x_phys[3] = x_norm[3] * 20.0
        else: # problem_id == 3
            # 使用变量代换强制满足时间间隔约束
            t_max = self.model.time_to_impact
            t_start = x_norm[2] * (t_max - 15) # 为后续间隔留出空间
            delta_t1 = x_norm[3] * 5.0 # 额外间隔1，范围[0, 5]秒
            delta_t2 = x_norm[4] * 5.0 # 额外间隔2，范围[0, 5]秒
            
            x_phys[2] = t_start
            x_phys[3] = t_start + self.cfg.MIN_LAUNCH_INTERVAL + delta_t1
            x_phys[4] = x_phys[3] + self.cfg.MIN_LAUNCH_INTERVAL + delta_t2
            
            # 解码起爆延迟
            x_phys[5] = x_norm[5] * 20.0
            x_phys[6] = x_norm[6] * 20.0
            x_phys[7] = x_norm[7] * 20.0
            
        return x_phys

    def cost_function(self, x_norm: np.ndarray) -> float:
        """优化器调用的成本函数。"""
        x_phys = self.decode(x_norm)
        if self.problem_id == 2:
            return self.model.cost_function_q2(x_phys)
        else:
            # 检查解码后的时间是否超出最大时间，这可能因为delta累加导致
            if x_phys[4] > self.model.time_to_impact - 5:
                return 200.0 # 惩罚
            return self.model.cost_function_q3(x_phys)

def solve_problem_1():
    # ... (此函数无需改动，保持原样) ...
    print("\n--- 开始求解问题 1 (解析法精确计算) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    
    uav_speed = 120.0
    fy1_pos = cfg.UAV_INITIAL_POS['FY1']
    target_pos = cfg.P_FALSE_TARGET
    direction = target_pos[:2] - fy1_pos[:2]
    uav_theta = np.arctan2(direction[1], direction[0])
    launch_time = 1.5
    det_delay = 3.6

    solution_vector = [uav_speed, uav_theta, launch_time, det_delay]
    details = model.get_final_details('FY1', solution_vector)
    shielding_time = details["total_shielding_time"]
    
    print("计算完成！")
    print(f"烟幕弹对 M1 的有效遮蔽时长: {shielding_time:.8f} s")
    
    return {"problem_id": 1, "max_shielding_time": shielding_time, "details": details}


def solve_problem_2():
    """问题二：单无人机单弹药策略优化 (增强版)。"""
    print("\n--- 开始求解问题 2 (归一化 + 重启策略) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    wrapper = OptimizationWrapper(model, 2)
    
    best_ever_solution = None
    best_ever_cost = float('inf')
    num_restarts = 3 # 执行3次重启搜索

    for i in range(num_restarts):
        print(f"\n--- 第 {i+1}/{num_restarts} 轮优化 ---")
        initial_guess_norm = np.random.rand(4)
        sigma0 = 0.2
        options = {
            'bounds': wrapper.bounds_norm, 
            'maxfevals': 5000, # 每次重启的评估次数
            'popsize': 40, 
            'verbose': -9
        }
        
        es = cma.CMAEvolutionStrategy(initial_guess_norm, sigma0, options)
        es.optimize(wrapper.cost_function)
        
        if es.result.fbest < best_ever_cost:
            best_ever_cost = es.result.fbest
            best_ever_solution = es.result.xbest
            print(f"发现新的最优解！成本: {best_ever_cost:.4f}")

    best_solution_phys = wrapper.decode(best_ever_solution)
    final_details = model.get_final_details('FY1', best_solution_phys)
    final_shielding_time = final_details["total_shielding_time"]

    print("\n" + "="*20 + " 问题二最终结果 " + "="*20)
    print("优化完成！")
    v, th, t1, dt1 = best_solution_phys
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"最大遮蔽时长: {final_shielding_time:.4f} s")

    return {"problem_id": 2, "max_shielding_time": final_shielding_time, "details": final_details}

def solve_problem_3():
    """问题三：单无人机三弹药策略优化 (增强版)。"""
    print("\n--- 开始求解问题 3 (归一化 + 变量代换 + 详细日志) ---")
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    wrapper = OptimizationWrapper(model, 3)
    
    initial_guess_norm = np.random.rand(8)
    sigma0 = 0.3
    options = {
        'bounds': wrapper.bounds_norm, 
        'maxfevals': 30000, 
        'popsize': 60, # 更高维度，更大种群
        'seed': int(time.time()),
        'verbose': -9
    }
    
    es = cma.CMAEvolutionStrategy(initial_guess_norm, sigma0, options)
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s. 8维归一化空间优化已启动...")
    
    log_interval = 20
    print("\n--- 优化过程追踪 ---")
    header = f"{'迭代':>5s} | {'评估数':>8s} | {'最优成本':>12s} | {'总遮蔽时长':>12s} | {'弹1 Pxy':>10s} | {'弹2 Pxy':>10s} | {'弹3 Pxy':>10s}"
    print(header)
    print("-" * len(header))

    while not es.stop():
        solutions_norm = es.ask()
        costs = [wrapper.cost_function(s) for s in solutions_norm]
        es.tell(solutions_norm, costs)
        
        if es.countiter % log_interval == 0:
            best_sol_norm = es.result.xbest
            best_sol_phys = wrapper.decode(best_sol_norm)
            
            metrics = model._calculate_combined_metrics('FY1', best_sol_phys[0], best_sol_phys[1], best_sol_phys[2:5], best_sol_phys[5:8])
            
            total_time = metrics["total_time"]
            proxies = metrics["proxy_costs"]
            
            print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | {total_time:11.4f}s | "
                  f"{proxies[0]:9.2e} | {proxies[1]:9.2e} | {proxies[2]:9.2e}")

    best_solution_norm = es.result.xbest
    best_solution_phys = wrapper.decode(best_solution_norm)
    final_details = model.get_final_details('FY1', best_solution_phys)
    final_shielding_time = final_details["total_shielding_time"]

    print("\n" + "="*20 + " 问题三最终结果 " + "="*20)
    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution_phys
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"  弹2: 投放时间={t2:.2f}s, 延迟={dt2:.2f}s")
    print(f"  弹3: 投放时间={t3:.2f}s, 延迟={dt3:.2f}s")
    print(f"最大遮蔽时长: {final_shielding_time:.4f} s")

    return {"problem_id": 3, "max_shielding_time": final_shielding_time, "details": final_details, "log": es}