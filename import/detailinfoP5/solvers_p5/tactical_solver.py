# solvers_p5/tactical_solver.py

import numpy as np
import cma
import time
from typing import List, Tuple, Dict

from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical
from .optimization_wrapper_p5 import P5OptimizationWrapper

def solve_tactical_problem(
    uav_id: str, 
    task_list: List[Tuple[int, str]], 
    solver_params: Dict
) -> Dict:
    """
    问题五的下层战术求解器。
    为单个无人机规划最优飞行与投弹策略以完成分配的任务。

    Args:
        uav_id: 无人机ID。
        task_list: 任务清单, e.g., [(0, 'M1'), (2, 'M2')]。
        solver_params: CMA-ES的控制参数, e.g., {'maxfevals': 10000, 'popsize': 40}。

    Returns:
        一个包含详细结果的字典。
    """
    # --- 1. 处理边缘情况 ---
    if not task_list:
        return {
            "objective_contributions": {'M1': 0.0, 'M2': 0.0, 'M3': 0.0},
            "scalar_cost": 0.0,
            "smoke_details": [],
            "status": "NO_TASK"
        }

    # --- 2. 初始化 ---
    cfg = Config()
    # 初始化模型时，初始missile_id不重要，因为评估函数会动态切换
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)
    wrapper = P5OptimizationWrapper(model, uav_id, task_list)

    # --- 3. 运行CMA-ES优化 ---
    # 这里可以加入重启逻辑，但为简化，先运行一次
    initial_guess_norm = np.random.rand(wrapper.dim)
    sigma0 = 0.3
    options = {
        'bounds': wrapper.bounds_norm,
        'maxfevals': solver_params.get('maxfevals', 10000),
        'popsize': solver_params.get('popsize', 40),
        'verbose': -9,
        'seed': int(time.time() * 1000) % (2**32 - 1)
    }
    
    es = cma.CMAEvolutionStrategy(initial_guess_norm, sigma0, options)
    
    # (可选) 添加迭代日志
    # while not es.stop():
    #     ...
    
    es.optimize(wrapper.cost_function)

    # --- 4. 获取并评估最终结果 ---
    best_solution_norm = es.result.xbest
    best_solution_phys = wrapper.decode(best_solution_norm)
    
    final_details = model.evaluate_single_uav_multi_target(uav_id, best_solution_phys, task_list)
    final_details["status"] = "OPTIMIZED"
    
    return final_details