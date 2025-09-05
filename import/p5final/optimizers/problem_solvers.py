# optimizers/problem_solvers.py

import numpy as np
import cma
import time
from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical

class OptimizationWrapper:
    """
    一个封装了变量归一化和约束处理的优化接口。
    优化器将在这个类提供的无约束、归一化的空间中进行操作，
    这极大地提升了优化算法的性能和稳定性。
    """
    def __init__(self, model: PhysicsModelAnalytical, problem_id: int):
        self.model = model
        self.cfg = model.config
        self.problem_id = problem_id
        
        # 根据问题ID确定决策向量的维度
        if self.problem_id == 2:
            self.dim = 4
        elif self.problem_id == 3:
            self.dim = 8
        elif self.problem_id == 4:
            self.dim = 12
        else:
            raise ValueError(f"不支持的问题ID: {problem_id}")
            
        self.bounds_norm = [[0.0] * self.dim, [1.0] * self.dim]

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        """将优化器操作的归一化向量 [0, 1] 解码为真实的物理参数向量。"""
        x_phys = np.zeros_like(x_norm)
        
        if self.problem_id == 2:
            # 解码: [v, θ, t_launch, Δt_det]
            x_phys[0] = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
            x_phys[1] = x_norm[1] * 2 * np.pi
            x_phys[2] = x_norm[2] * (self.model.time_to_impact - 5) # 投放时间
            x_phys[3] = x_norm[3] * 20.0 # 引爆延迟

        elif self.problem_id == 3:
            # 解码: [v, θ, t1, t2, t3, Δt1, Δt2, Δt3]
            x_phys[0] = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
            x_phys[1] = x_norm[1] * 2 * np.pi
            
            # 使用变量代换技巧来保证 t1 < t2 < t3 且间隔满足要求
            t_max = self.model.time_to_impact
            t_start = x_norm[2] * (t_max - 15) # 第一个投放时间的基础值
            delta_t1 = x_norm[3] * 5.0 # 第1和第2个投放时间的附加间隔
            delta_t2 = x_norm[4] * 5.0 # 第2和第3个投放时间的附加间隔
            
            x_phys[2] = t_start
            x_phys[3] = t_start + self.cfg.MIN_LAUNCH_INTERVAL + delta_t1
            x_phys[4] = x_phys[3] + self.cfg.MIN_LAUNCH_INTERVAL + delta_t2
            
            x_phys[5:8] = x_norm[5:8] * 20.0 # 三个引爆延迟

        elif self.problem_id == 4:
            # 解码: 3架无人机，每架 [v, θ, t_launch, Δt_det]
            for i in range(3):
                offset = i * 4
                x_phys[offset]     = self.cfg.V_UAV_MIN + x_norm[offset] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
                x_phys[offset + 1] = x_norm[offset + 1] * 2 * np.pi
                x_phys[offset + 2] = x_norm[offset + 2] * (self.model.time_to_impact - 5)
                x_phys[offset + 3] = x_norm[offset + 3] * 20.0
            
        return x_phys

    def cost_function(self, x_norm: np.ndarray) -> float:
        """供优化器调用的成本函数，接收归一化向量。"""
        x_phys = self.decode(x_norm)
        
        # 根据问题ID调用对应的物理模型成本函数
        if self.problem_id == 2:
            return self.model.cost_function_q2(x_phys)
        elif self.problem_id == 3:
            # 快速检查硬约束，避免无效计算
            if x_phys[4] > self.model.time_to_impact - 5: return 200.0
            return self.model.cost_function_q3(x_phys)
        elif self.problem_id == 4:
            return self.model.cost_function_q4(x_phys)


def solve_problem_1():
    """问题一：固定策略的精确计算。"""
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
    """通用的带重启的优化执行器，内置详细日志功能。"""
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
        
        # --- 详细日志模块 ---
        log_interval = 20
        if problem_id == 3 or problem_id == 4:
            print("\n" + "="*110)
            title = "问题三优化追踪 (FY1, 3弹)" if problem_id == 3 else "问题四优化追踪 (FY1, FY2, FY3 协同)"
            print(title.center(110))
            
            if problem_id == 3:
                header = (f"{'迭代':>5s} | {'评估数':>8s} | {'总成本':>12s} | {'总时长(s)':>10s} | "
                          f"{'弹1时长(s)':>10s} | {'弹2时长(s)':>10s} | {'弹3时长(s)':>10s} | "
                          f"{'弹1代理成本':>12s} | {'弹2代理成本':>12s} | {'弹3代理成本':>12s}")
            else: # problem_id == 4
                header = (f"{'迭代':>5s} | {'评估数':>8s} | {'总成本':>12s} | {'总时长(s)':>10s} | "
                          f"{'FY1时长(s)':>10s} | {'FY2时长(s)':>10s} | {'FY3时长(s)':>10s} | "
                          f"{'FY1代理成本':>12s} | {'FY2代理成本':>12s} | {'FY3代理成本':>12s}")
            print(header)
            print("-" * len(header))

        while not es.stop():
            solutions_norm = es.ask()
            costs = [wrapper.cost_function(s) for s in solutions_norm]
            es.tell(solutions_norm, costs)
            
            if (problem_id == 3 or problem_id == 4) and es.countiter % log_interval == 0:
                best_sol_norm = es.result.xbest
                best_sol_phys = wrapper.decode(best_sol_norm)
                
                # 分别计算每个烟幕的详细指标
                if problem_id == 3:
                    metrics = model._calculate_combined_metrics('FY1', best_sol_phys[0], best_sol_phys[1], best_sol_phys[2:5], best_sol_phys[5:8])
                    total_time = metrics['total_time']
                    proxies = metrics['proxy_costs']
                    
                    # 需要重新计算单弹时长
                    individual_times = []
                    for j in range(3):
                        p_det, t_det = model._get_detonation_params('FY1', best_sol_phys[0], best_sol_phys[1], best_sol_phys[2+j], best_sol_phys[5+j])
                        m = model._calculate_metrics_for_one_smoke(p_det, t_det)
                        individual_times.append(sum(e - s for s, e in m['intervals']))
                else: # problem_id == 4
                    uav_ids = ['FY1', 'FY2', 'FY3']
                    strategies = np.reshape(best_sol_phys, (3, 4))
                    metrics = model._calculate_multi_uav_metrics(uav_ids, strategies)
                    total_time = metrics['total_time']
                    proxies = metrics['proxy_costs']
                    
                    individual_times = []
                    for j, uav_id in enumerate(uav_ids):
                        p_det, t_det = model._get_detonation_params(uav_id, *strategies[j])
                        m = model._calculate_metrics_for_one_smoke(p_det, t_det)
                        individual_times.append(sum(e - s for s, e in m['intervals']))

                print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | "
                      f"{total_time:10.4f} | {individual_times[0]:10.4f} | {individual_times[1]:10.4f} | {individual_times[2]:10.4f} | "
                      f"{proxies[0]:12.2e} | {proxies[1]:12.2e} | {proxies[2]:12.2e}")

        if es.result.fbest < best_ever_cost:
            best_ever_cost = es.result.fbest
            best_ever_solution_norm = es.result.xbest
            print(f"本轮优化结束，发现新的全局最优解！成本: {best_ever_cost:.6f}")

    best_solution_phys = wrapper.decode(best_ever_solution_norm)
    return best_solution_phys, model


def solve_problem_2():
    """问题二：单无人机单弹药策略优化。"""
    print("\n--- 开始求解问题 2 (归一化 + 重启策略) ---")
    
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=2, num_restarts=3, maxfevals_per_restart=5000, popsize=40
    )

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
    """问题三：单无人机三弹药策略优化。"""
    print("\n--- 开始求解问题 3 (归一化 + 变量代换 + 重启 + 详细日志) ---")
    
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=3, num_restarts=2, maxfevals_per_restart=20000, popsize=60
    )

    final_details = model.get_final_details('FY1', best_solution_phys)
    final_shielding_time = final_details["total_shielding_time"]

    print("\n" + "="*25 + " 问题三最终最优策略详解 " + "="*25)
    print("优化完成！")
    v, th, t1, t2, t3, dt1, dt2, dt3 = best_solution_phys
    print(f"最优策略: 速度={v:.2f} m/s, 方向={np.rad2deg(th):.2f}°")
    
    # 详细打印每枚弹药的信息
    launch_times = [t1, t2, t3]
    det_delays = [dt1, dt2, dt3]
    for i in range(3):
        print(f"\n  - 弹药 {i+1}:")
        print(f"    投放时间: {launch_times[i]:.2f}s, 引爆延迟: {det_delays[i]:.2f}s")
        p_det, t_det = model._get_detonation_params('FY1', v, th, launch_times[i], det_delays[i])
        metrics = model._calculate_metrics_for_one_smoke(p_det, t_det)
        individual_time = sum(end - start for start, end in metrics['intervals'])
        interval_str = ", ".join([f"[{s:.2f}, {e:.2f}]" for s, e in metrics['intervals']])
        print(f"    独立遮蔽时长: {individual_time:.4f} s")
        print(f"    遮蔽时间区间: {interval_str if interval_str else '无'}")

    print("\n--- 协同结果汇总 ---")
    print(f"最终协同总时长: {final_shielding_time:.4f} s")
    print(f"(注: 总时长不等于简单相加，因存在重叠)")

    return {"problem_id": 3, "max_shielding_time": final_shielding_time, "details": final_details}


def solve_problem_4():
    """问题四：三无人机协同策略优化。"""
    print("\n--- 开始求解问题 4 (多智能体协同优化) ---")
    
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=4, num_restarts=3, maxfevals_per_restart=30000, popsize=60
    )
    
    uav_ids = ['FY1', 'FY2', 'FY3']
    final_details = model.get_final_details_multi_uav(uav_ids, best_solution_phys)
    total_shielding_time = final_details["total_shielding_time"]

    print("\n" + "="*25 + " 问题四最终最优策略详解 " + "="*25)
    print("优化完成！")
    
    strategies = np.reshape(best_solution_phys, (3, 4))
    for i, uav_id in enumerate(uav_ids):
        v, th, t, dt = strategies[i]
        print(f"\n  - {uav_id} 策略:")
        print(f"    速度: {v:.2f} m/s, 方向: {np.rad2deg(th):.2f}°")
        print(f"    投放时间: {t:.2f}s, 引爆延迟: {dt:.2f}s")
        
        p_det, t_det = model._get_detonation_params(uav_id, v, th, t, dt)
        metrics = model._calculate_metrics_for_one_smoke(p_det, t_det)
        individual_time = sum(end - start for start, end in metrics['intervals'])
        interval_str = ", ".join([f"[{s:.2f}, {e:.2f}]" for s, e in metrics['intervals']])
        print(f"    独立遮蔽时长: {individual_time:.4f} s")
        print(f"    遮蔽时间区间: {interval_str if interval_str else '无'}")

    print("\n--- 协同结果汇总 ---")
    merged_intervals = final_details.get("merged_intervals", []) # 假设 get_final_details_multi_uav 会返回这个
    if not merged_intervals: # 如果没返回，我们在这里计算
        all_intervals = []
        for event in final_details['smoke_events']:
            metrics = model._calculate_metrics_for_one_smoke(event['p_det'], event['t_det'])
            all_intervals.extend(metrics['intervals'])
        if all_intervals:
            all_intervals.sort(key=lambda x: x[0])
            merged_intervals = [all_intervals[0]]
            for start, end in all_intervals[1:]:
                if start <= merged_intervals[-1][1]:
                    merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
                else:
                    merged_intervals.append((start, end))

    final_interval_str = ", ".join([f"[{s:.2f}, {e:.2f}]" for s, e in merged_intervals])
    print(f"最终协同遮蔽区间: {final_interval_str if final_interval_str else '无'}")
    print(f"最终协同总时长: {total_shielding_time:.4f} s")

    return {"problem_id": 4, "max_shielding_time": total_shielding_time, "details": final_details}