# solvers_for_p5/p5_components.py

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

class AssignmentProblem(ElementwiseProblem):
    def __init__(self, parallel_evaluator, tactical_solver_params_func):
        self.evaluator = parallel_evaluator
        self.get_tactical_solver_params = tactical_solver_params_func
        
        super().__init__(n_var=15, n_obj=3, n_constr=5, xl=0, xu=3, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        assignment_matrix = x.reshape(5, 3)
        
        # 1. 计算约束
        constraints = [np.sum(row) - 3 for row in assignment_matrix]
        out["G"] = np.array(constraints)

        # 2. 获取当前阶段的下层求解器参数
        # algorithm对象通过kwargs传入，可以获取当前代数
        algorithm = kwargs.get("algorithm")
        current_gen = algorithm.n_gen if algorithm else 0
        solver_params = self.get_tactical_solver_params(current_gen)

        # 3. 并行评估
        objectives = self.evaluator.evaluate(assignment_matrix, solver_params)
        
        # 4. 设置目标值 (最小化负时长)
        out["F"] = -objectives

# --- 自定义初始化、交叉、变异 (确保约束满足) ---

class SmartInitialization(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            # 伪代码：实现启发式和随机初始化
            # ...
            X[i, :] = np.random.randint(0, 4, size=problem.n_var) # 简化版
        return X

class ConstrainedCrossover(Crossover):
    # ... 实现交叉逻辑，并修复行和约束 ...
    def __init__(self):
        super().__init__(2, 2) # 2个父代，2个子代

    def _do(self, problem, X, **kwargs):
        # 简化版：单点交叉后不修复
        _, n_mat_cols = X.shape
        Y = np.full_like(X, -1, dtype=int)
        for k in range(0, X.shape[0], 2):
            p1, p2 = X[k], X[k+1]
            crossover_point = np.random.randint(1, n_mat_cols)
            Y[k, :crossover_point] = p1[:crossover_point]
            Y[k, crossover_point:] = p2[crossover_point:]
            Y[k+1, :crossover_point] = p2[:crossover_point]
            Y[k+1, crossover_point:] = p1[crossover_point:]
        return Y

class ConstrainedMutation(Mutation):
    # ... 实现变异逻辑，并修复行和约束 ...
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            if np.random.rand() < 0.1: # 10%的变异率
                idx = np.random.randint(0, problem.n_var)
                X[i, idx] = np.random.randint(0, 4)
        return X