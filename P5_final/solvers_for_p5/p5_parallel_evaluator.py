# solvers_for_p5/p5_parallel_evaluator.py

import numpy as np
import multiprocessing
from optimizers.problem_solvers import _run_optimization_with_restarts # 复用下层优化逻辑
from models.physics_model_analytical import PhysicsModelAnalytical
from config import Config

# 全局变量，用于在多进程中共享模型实例，避免重复初始化
model_instances = {}

def init_worker(missile_id):
    """每个工作进程的初始化函数"""
    global model_instances
    if missile_id not in model_instances:
        model_instances[missile_id] = PhysicsModelAnalytical(missile_id=missile_id, config_obj=Config())

def solve_tactical_problem_worker(args):
    """可被并行调用的工作函数"""
    missile_id, assignment_for_missile, solver_params = args
    
    # 这里需要一个能处理动态无人机和弹药分配的下层优化器
    # 为简化，我们暂时假设一个函数 solve_dynamic_tactical(...) 存在
    # 它的逻辑会和 problem_solvers.py 中的类似，但更通用
    # 返回值是 (最大遮蔽时长, 最优策略向量)
    
    # 伪代码实现，实际需要您根据 problem_solvers.py 泛化
    # from .p5_tactical_solver import solve_dynamic_tactical
    # max_time, _ = solve_dynamic_tactical(missile_id, assignment_for_missile, solver_params)
    
    # 临时替代实现：返回一个基于弹药数量的估计值
    num_smokes = sum(assignment_for_missile.values())
    max_time = num_smokes * 5.0 - np.random.rand() # 模拟计算
    
    return max_time

class ParallelEvaluator:
    def __init__(self, cpu_cores):
        num_cores = cpu_cores if cpu_cores > 0 else multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(processes=num_cores)
        self.missile_ids = ['M1', 'M2', 'M3']

    def evaluate(self, assignment_matrix, solver_params):
        """接收一个分配矩阵，并行计算三个目标值"""
        tasks = []
        for j, missile_id in enumerate(self.missile_ids):
            assignment_for_missile = {}
            for i in range(5): # 5架无人机
                if assignment_matrix[i, j] > 0:
                    uav_id = f"FY{i+1}"
                    assignment_for_missile[uav_id] = assignment_matrix[i, j]
            
            if assignment_for_missile: # 如果有无人机被分配
                tasks.append((missile_id, assignment_for_missile, solver_params))
            else: # 如果没有无人机，时长为0
                tasks.append(None)

        results = []
        # 伪并行调用，实际需要一个真正的下层求解器
        for task in tasks:
            if task:
                results.append(solve_tactical_problem_worker(task))
            else:
                results.append(0.0)
        
        return np.array(results)

    def close(self):
        self.pool.close()
        self.pool.join()