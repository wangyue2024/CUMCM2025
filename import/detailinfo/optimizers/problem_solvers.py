# optimizers/problem_solvers.py

import numpy as np
import cma
import time
# ！！！关键修正：从 typing 模块导入 Dict 和 List ！！！
from typing import Dict, List

from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical

# ... OptimizationWrapper 类的代码保持不变 ...
class OptimizationWrapper:
    def __init__(self, model: PhysicsModelAnalytical, problem_id: int):
        self.model = model
        self.cfg = model.config
        self.problem_id = problem_id
        if self.problem_id == 2:
            self.dim = 4
        elif self.problem_id == 3:
            self.dim = 8
        elif self.problem_id == 4:
            self.dim = 12
        self.bounds_norm = [[0.0] * self.dim, [1.0] * self.dim]

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        x_phys = np.zeros_like(x_norm)
        
        if self.problem_id == 2 or self.problem_id == 3:
            x_phys[0] = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
            x_phys[1] = x_norm[1] * 2 * np.pi
            if self.problem_id == 2:
                x_phys[2] = x_norm[2] * (self.model.time_to_impact - 5)
                x_phys[3] = x_norm[3] * 20.0
            else: # problem_id == 3
                t_max = self.model.time_to_impact
                t_start = x_norm[2] * (t_max - 15)
                delta_t1 = x_norm[3] * 5.0
                delta_t2 = x_norm[4] * 5.0
                x_phys[2] = t_start
                x_phys[3] = t_start + self.cfg.MIN_LAUNCH_INTERVAL + delta_t1
                x_phys[4] = x_phys[3] + self.cfg.MIN_LAUNCH_INTERVAL + delta_t2
                x_phys[5:8] = x_norm[5:8] * 20.0

        elif self.problem_id == 4:
            for i in range(3):
                offset = i * 4
                x_phys[offset] = self.cfg.V_UAV_MIN + x_norm[offset] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
                x_phys[offset + 1] = x_norm[offset + 1] * 2 * np.pi
                x_phys[offset + 2] = x_norm[offset + 2] * (self.model.time_to_impact - 5)
                x_phys[offset + 3] = x_norm[offset + 3] * 20.0
            
        return x_phys

    def cost_function(self, x_norm: np.ndarray) -> float:
        x_phys = self.decode(x_norm)
        if self.problem_id == 2:
            return self.model.cost_function_q2(x_phys)
        elif self.problem_id == 3:
            if x_phys[4] > self.model.time_to_impact - 5: return 200.0
            return self.model.cost_function_q3(x_phys)
        elif self.problem_id == 4:
            for i in range(3):
                if x_phys[i*4 + 2] > self.model.time_to_impact - 5:
                    return 200.0
            return self.model.cost_function_q4(x_phys)


def _print_final_report(details: Dict):
    """根据 get_full_details_for_report 返回的字典，打印标准格式的最终报告。"""
    # ... 函数的其余部分完全不变 ...
    print("\n" + "="*30 + " 最终策略详情 " + "="*30)
    
    header = (
        f"{'无人机':<6s} | {'方向(°)':>8s} | {'速度(m/s)':>10s} | {'干扰弹':>6s} | "
        f"{'投放时间(s)':>12s} | {'投放点 (x,y,z)':>25s} | "
        f"{'爆炸时间(s)':>12s} | {'爆炸点 (x,y,z)':>25s} | "
        f"{'独立时长(s)':>12s} | {'目标导弹':>8s}"
    )
    print(header)
    print("-" * len(header))

    for smoke in details["smoke_details"]:
        lp = smoke['launch_point']
        dp = smoke['detonation_point']
        row = (
            f"{smoke['uav_id']:<6s} | {smoke['uav_direction_deg']:>8.2f} | {smoke['uav_speed']:>10.2f} | {smoke['smoke_id']:>6s} | "
            f"{smoke['launch_time']:>12.4f} | {f'({lp[0]:.1f}, {lp[1]:.1f}, {lp[2]:.1f})':>25s} | "
            f"{smoke['detonation_time']:>12.4f} | {f'({dp[0]:.1f}, {dp[1]:.1f}, {dp[2]:.1f})':>25s} | "
            f"{smoke['individual_shielding_time']:>12.4f} | {smoke['target_missile_id']:>8s}"
        )
        print(row)
        
        intervals_str = ", ".join([f"[{start:.2f}s, {end:.2f}s]" for start, end in smoke['shielding_intervals']])
        if not intervals_str:
            intervals_str = "无"
        print(f"  └─ 遮蔽区间: {intervals_str}")

    print("-" * len(header))
    total_time = details['total_shielding_time']
    merged_intervals_str = ", ".join([f"[{start:.2f}s, {end:.2f}s]" for start, end in details['merged_intervals']])
    if not merged_intervals_str:
        merged_intervals_str = "无"
    print(f"\n>>> 最终合并总有效遮蔽时长: {total_time:.4f} 秒")
    print(f"    合并后总遮蔽区间: {merged_intervals_str}")
    print("="*75)


# ... (_run_optimization_with_restarts, solve_problem_1, 2, 3, 4 的代码保持不变) ...
def _run_optimization_with_restarts(problem_id: int, num_restarts: int, maxfevals_per_restart: int, popsize: int):
    """通用的带重启的优化执行器 (带详细日志输出)。"""
    cfg = Config()
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    wrapper = OptimizationWrapper(model, problem_id)
    
    uav_ids = []
    if problem_id in [2, 3]:
        uav_ids = ['FY1']
    elif problem_id == 4:
        uav_ids = ['FY1', 'FY2', 'FY3']

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
        if problem_id in [3, 4]:
            print("\n--- 优化过程追踪 ---")
            num_smokes = 3
            header_s_times = " | ".join([f"S{j+1}(s)" for j in range(num_smokes)])
            header_s_proxies = " | ".join([f"Pxy{j+1}" for j in range(num_smokes)])
            header = f"{'Iter':>5s} | {'Evals':>8s} | {'Cost':>12s} | {'Total(s)':>9s} | {header_s_times} | {header_s_proxies}"
            print(header)
            print("-" * len(header))

        while not es.stop():
            solutions_norm = es.ask()
            costs = [wrapper.cost_function(s) for s in solutions_norm]
            es.tell(solutions_norm, costs)
            
            if es.countiter % log_interval == 0 and problem_id in [3, 4]:
                best_sol_norm = es.result.xbest
                best_sol_phys = wrapper.decode(best_sol_norm)
                
                details = model.get_full_details_for_report(uav_ids, best_sol_phys)
                
                s_times_str = " | ".join([f"{s['individual_shielding_time']:6.4f}" for s in details['smoke_details']])
                s_proxies_str = " | ".join([f"{s['proxy_cost']:8.2e}" for s in details['smoke_details']])
                
                print(f"{es.countiter:5d} | {es.countevals:8d} | {es.result.fbest:12.6f} | "
                      f"{details['total_shielding_time']:8.4f} | {s_times_str} | {s_proxies_str}")

        if es.result.fbest < best_ever_cost:
            best_ever_cost = es.result.fbest
            best_ever_solution_norm = es.result.xbest
            print(f"本轮优化结束，发现新的全局最优解！成本: {best_ever_cost:.6f}")

    best_solution_phys = wrapper.decode(best_ever_solution_norm)
    return best_solution_phys, model

def solve_problem_1():
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
    final_details = model.get_full_details_for_report(['FY1'], solution_vector)
    
    _print_final_report(final_details)
    return {"problem_id": 1, "details": final_details}

def solve_problem_2():
    print("\n--- 开始求解问题 2 (单弹药优化) ---")
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=2, num_restarts=3, maxfevals_per_restart=5000, popsize=40
    )
    final_details = model.get_full_details_for_report(['FY1'], best_solution_phys)
    _print_final_report(final_details)
    return {"problem_id": 2, "details": final_details}

def solve_problem_3():
    print("\n--- 开始求解问题 3 (三弹药协同优化) ---")
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=3, num_restarts=2, maxfevals_per_restart=20000, popsize=60
    )
    final_details = model.get_full_details_for_report(['FY1'], best_solution_phys)
    _print_final_report(final_details)
    return {"problem_id": 3, "details": final_details}

def solve_problem_4():
    print("\n--- 开始求解问题 4 (多智能体协同优化) ---")
    best_solution_phys, model = _run_optimization_with_restarts(
        problem_id=4, num_restarts=3, maxfevals_per_restart=30000, popsize=60
    )
    final_details = model.get_full_details_for_report(['FY1', 'FY2', 'FY3'], best_solution_phys)
    _print_final_report(final_details)
    return {"problem_id": 4, "details": final_details}