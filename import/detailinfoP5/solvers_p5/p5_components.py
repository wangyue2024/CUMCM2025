# solvers_p5/p5_components.py

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair

class AssignmentProblem(ElementwiseProblem):
    def __init__(self, evaluator):
        self.evaluator = evaluator
        # 15个决策变量 (5架无人机 * 3枚弹)
        # 变量范围 0-3 (0:不用, 1:打M1, 2:打M2, 3:打M3)
        # 3个目标 (最大化 T_M1, T_M2, T_M3)
        # 5个约束 (每架无人机最多用3枚弹)
        super().__init__(n_var=15, n_obj=3, n_constr=5, xl=0, xu=3, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. 计算约束
        # 约束定义为 g(x) <= 0
        constraints = []
        for i in range(5):
            num_used_smokes = np.sum(x[i*3 : i*3+3] > 0)
            constraints.append(num_used_smokes - 3)
        out["G"] = np.array(constraints)

        # 2. 调用评估器计算目标值
        # algorithm对象由pymoo在运行时传入，用于获取当前代数
        algorithm = kwargs.get("algorithm")
        objectives = self.evaluator.evaluate(x, algorithm)
        
        # 3. 设置目标值 (pymoo默认最小化，所以我们最小化负时长)
        out["F"] = -objectives

class AssignmentRepair(Repair):
    """修复算子，确保每个无人机使用的弹药数不超过3"""
    def _do(self, problem, pop, **kwargs):
        for ind in pop:
            for i in range(5):
                smokes = ind.X[i*3 : i*3+3]
                used_indices = np.where(smokes > 0)[0]
                
                while len(used_indices) > 3:
                    # 如果弹药超限，随机移除一个任务
                    remove_idx = np.random.choice(used_indices)
                    smokes[remove_idx] = 0 # 设为不使用
                    used_indices = np.where(smokes > 0)[0]
        return pop

class SmartInitialization(Sampling):
    """约束内的随机初始化"""
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            for uav_idx in range(5):
                # 为每架无人机随机分配0-3枚弹药
                num_to_use = np.random.randint(0, 4)
                if num_to_use > 0:
                    # 随机选择要使用的弹药槽位
                    smoke_indices = np.random.choice([0, 1, 2], size=num_to_use, replace=False)
                    # 为这些弹药随机分配目标
                    targets = np.random.randint(1, 4, size=num_to_use)
                    for smoke_idx, target in zip(smoke_indices, targets):
                        X[i, uav_idx*3 + smoke_idx] = target
        return X