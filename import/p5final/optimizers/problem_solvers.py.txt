# optimizers/problem_solvers.py

import numpy as np
import cma
import time
from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical

class OptimizationWrapper:
    # ... (这个类无需改动，保持原样) ...
    """
    一个封装了变量归一化和约束处理的优化接口。
    优化器将在这个类提供的无约束、归一化的空间中进行操作。
    """
    def __init__(self, model: PhysicsModelAnalytical, problem_id: int):
        self.model = model
        self.cfg = model.config
        self.problem_id = problem_id
        self.dim = 4 if self.problem_id == 2 else 8
        self.bounds_norm = [[0.0] * self.dim, [1.0] * self.dim]

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        """将归一化的向量解码为真实的物理参数向量。"""
        x_phys = np.zeros_like(x_norm)
        
        x_phys[0] = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
        x_phys[1] = x_norm[1] * 2 * np.pi

        if self.problem_id == 2:
            x_phys[2] = x_norm[2] * (self.model.time_to_impact - 5)
            x_phys[3] = x_norm[3] * 20.0
        else:
            t_max = self.model.time_to_impact
            t_start = x_norm[2] * (t_max - 15)
            delta_t1 = x_norm[3] * 5.0
            delta_t2 = x_norm[4] * 5.0
            
            x_phys[2] = t_start
            x_phys[3] = t_start + self.cfg.MIN_LAUNCH_INTERVAL + delta_t1
            x_phys[4] = x_phys[3] + self.cfg.MIN_LAUNCH_INTERVAL + delta_t2
            
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
            if x_phys[4] > self.model.time_to_impact - 5:
                return 200.0
            return self.model.cost_function_q3(x_phys)

def solve_problem_1():
    # ... (这个函数无需改动，保持原样) ...
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

def _run_optimization_with_restarts(problem_id: int, num_restarts: int, maxfevals_per_restart: int, popsize: int):
    """通用的带重启的优化执行器"""
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    wrapper = OptimizationWrapper(model, problem_id)
    
    best_ever_solution_norm = None
    best_ever_cost = float('inf')

    for i in range(num_restarts):
        print(f"\n--- 第 {i+1}/{num_restarts} 轮优化 ---")
        initial_guess_norm = np.random.rand(wrapper.dim)
        sigma0 = 0.3
        options = {
            'bounds': wrapper.bounds_norm, 
            'maxfevals': maxfevals_per_restart,
            'popsize': popsize, 
            'verbose': -9,
            'seed': int(time.time() + i)
        }
        
        es = cma.CMAEvolutionStrategy(initial_guess_norm, sigma0, options)
        
        log_interval = 20
        if problem_id == 3:
            print("\n--- 优化过程追踪 ---")
            header = f"{'Iter':>5s} | {'Evals':>8s} | {'Cost':>12s} | {'Total(s)':>9s} | {'S1(s)':>7s} | {'S2(s)':>7s} | {'S3(s)':>7s} | {'Pxy1':>9s} | {'Pxy2':>9s} | {'Pxy3':>9s}"
            print(header)
            print("-" * len(header))

        while not es.stop():
            solutions_norm = es.ask()
            costs = [wrapper.cost_function(s) for s in solutions_norm]
            es.tell(solutions_norm, costs)
            
            if es.countiter % log_interval == 0 and problem_id == 3:
                best_sol_norm = es.result.xbest
                best_sol_phys = wrapper.decode(best_sol_norm)
                
                metrics = model._calculate_combined_metrics('FY1', best_sol_phys[0], best_sol_phys[1], best_sol_phys[2:5], best_sol_phys[5:8])
                
                # ===================== FIX START =====================
                # 修正点 1: 显式传递 p_det 和 t_det
                individual_metrics = []
                for j in range(3):
                    # 先计算出单个烟幕弹的投放和引爆参数
                    p_uav_0 = cfg.UAV_INITIAL_POS['FY1']
                    v, th = best_sol_phys[0], best_sol_phys[1]
                    t, dt = best_sol_phys[2+j], best_sol_phys[5+j]
                    v_vec_uav = np.array([v * np.cos(th), v * np.sin(th), 0])
                    p_launch = p_uav_0 + v_vec_uav * t
                    p_det = p_launch + v_vec_uav * dt + 0.5 * np.array([0, 0, -cfg.G]) * dt**2
                    t_det = t + dt
                    
                    # 使用正确的参数调用
                    individual_metrics.append(model._calculate_metrics_for_one_smoke(p_det, t_det))
                # ===================== FIX END =======================

                s1_time = sum(end - start for start, end in individual_metrics[0]['intervals'])
                s2_time = sum(end - start for start, end in individual_metrics[1]['intervals'])
                s3_time = sum(end - start for start, end in individual_metrics[2]['intervals'])

                proxies = metrics["proxy_costs"]
                
                print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | "
                      f"{metrics['total_time']:8.4f} | {s1_time:6.4f} | {s2_time:6.4f} | {s3_time:6.4f} | "
                      f"{proxies[0]:8.2e} | {proxies[1]:8.2e} | {proxies[2]:8.2e}")

        if es.result.fbest < best_ever_cost:
            best_ever_cost = es.result.fbest
            best_ever_solution_norm = es.result.xbest
            print(f"本轮优化结束，发现新的全局最优解！成本: {best_ever_cost:.6f}")

    best_solution_phys = wrapper.decode(best_ever_solution_norm)
    final_details = model.get_final_details('FY1', best_solution_phys)
    final_shielding_time = final_details["total_shielding_time"]

    return best_solution_phys, final_shielding_time, final_details, model # 返回model实例以便后续使用

def solve_problem_2():
    """问题二：单无人机单弹药策略优化 (增强版)。"""
    print("\n--- 开始求解问题 2 (归一化 + 重启策略) ---")
    
    best_solution_phys, final_shielding_time, final_details, _ = _run_optimization_with_restarts(
        problem_id=2,
        num_restarts=3,
        maxfevals_per_restart=5000,
        popsize=40
    )

    print("\n" + "="*20 + " 问题二最终结果 " + "="*20)
    print("优化完成！")
    v, th, t1, dt1 = best_solution_phys
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"最大遮蔽时长: {final_shielding_time:.4f} s")

    return {"problem_id": 2, "max_shielding_time": final_shielding_time, "details": final_details}

def solve_problem_3():
    """问题三：单无人机三弹药策略优化 (终极版)。"""
    print("\n--- 开始求解问题 3 (归一化 + 变量代换 + 重启 + 详细日志) ---")
    
    best_solution_phys, final_shielding_time, final_details, model = _run_optimization_with_restarts(
        problem_id=3,
        num_restarts=2,
        maxfevals_per_restart=20000,
        popsize=60
    )

    print("\n" + "="*20 + " 问题三最终结果 " + "="*20)
    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution_phys
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    print(f"  弹1: 投放时间={t1:.2f}s, 延迟={dt1:.2f}s")
    print(f"  弹2: 投放时间={t2:.2f}s, 延迟={dt2:.2f}s")
    print(f"  弹3: 投放时间={t3:.2f}s, 延迟={dt3:.2f}s")
    print(f"最大遮蔽时长: {final_shielding_time:.4f} s")

    # ===================== FIX START =====================
    # 修正点 2: 同样使用显式参数传递
    cfg = Config()
    p_uav_0 = cfg.UAV_INITIAL_POS['FY1']
    v_vec_uav = np.array([v * np.cos(th), v * np.sin(th), 0])
    
    individual_metrics = []
    for t, dt in zip([t1, t2, t3], [dt1, dt2, dt3]):
        p_launch = p_uav_0 + v_vec_uav * t
        p_det = p_launch + v_vec_uav * dt + 0.5 * np.array([0, 0, -cfg.G]) * dt**2
        t_det = t + dt
        individual_metrics.append(model._calculate_metrics_for_one_smoke(p_det, t_det))
    # ===================== FIX END =======================

    s1_time = sum(end - start for start, end in individual_metrics[0]['intervals'])
    s2_time = sum(end - start for start, end in individual_metrics[1]['intervals'])
    s3_time = sum(end - start for start, end in individual_metrics[2]['intervals'])
    print(f"  - 单独贡献: 弹1={s1_time:.4f}s, 弹2={s2_time:.4f}s, 弹3={s3_time:.4f}s")
    print(f"  - (注: 总时长不等于简单相加，因存在重叠)")

    return {"problem_id": 3, "max_shielding_time": final_shielding_time, "details": final_details}