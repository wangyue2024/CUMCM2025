# solvers_for_p5/p5_solver.py

import os
import pickle
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from .p5_config import P5Config
from .p5_utils import setup_logger
from .p5_components import AssignmentProblem, SmartInitialization, ConstrainedCrossover, ConstrainedMutation
from .p5_parallel_evaluator import ParallelEvaluator

logger = setup_logger()

class CheckpointCallback(Callback):
    """每代结束时保存快照的回调函数"""
    def __init__(self):
        super().__init__()
        self.n_gen = 0

    def notify(self, algorithm):
        self.n_gen += 1
        if self.n_gen % 10 == 0: # 每10代保存一次
            with open(P5Config.CHECKPOINT_FILE, "wb") as f:
                pickle.dump(algorithm, f)
            logger.info(f"已在第 {self.n_gen} 代保存快照到 {P5Config.CHECKPOINT_FILE}")

def get_tactical_solver_params(current_gen):
    """根据当前代数返回对应的下层求解器精度"""
    total_gens = P5Config.N_GENERATIONS
    for stage_ratio, params in P5Config.TACTICAL_SOLVER_STAGES:
        if current_gen / total_gens <= stage_ratio:
            return params
    return P5Config.TACTICAL_SOLVER_STAGES[-1][1]

def solve_problem_5():
    """问题五的主求解函数"""
    logger.info("="*50)
    logger.info("开始求解问题五：多无人机多目标协同策略")
    logger.info("="*50)

    start_time = time.time()

    # 1. 初始化并行评估器
    evaluator = ParallelEvaluator(cpu_cores=P5Config.CPU_CORES)

    # 2. 实例化问题
    problem = AssignmentProblem(evaluator, get_tactical_solver_params)

    # 3. 检查是否有断点文件，实现续跑
    if os.path.exists(P5Config.CHECKPOINT_FILE):
        with open(P5Config.CHECKPOINT_FILE, "rb") as f:
            algorithm = pickle.load(f)
        logger.info(f"成功从 {P5Config.CHECKPOINT_FILE} 加载算法状态，继续运行。")
    else:
        algorithm = NSGA2(
            pop_size=P5Config.POP_SIZE,
            sampling=SmartInitialization(),
            crossover=ConstrainedCrossover(),
            mutation=ConstrainedMutation(),
            eliminate_duplicates=True
        )
        logger.info("未找到快照文件，开始新的优化任务。")

    # 4. 运行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', P5Config.N_GENERATIONS),
        callback=CheckpointCallback(),
        seed=1,
        verbose=True,
        save_history=True # 保存历史记录以供分析
    )

    evaluator.close()
    end_time = time.time()
    logger.info(f"问题五求解完成，总耗时: {(end_time - start_time) / 3600:.2f} 小时")

    # 5. 保存并分析最终结果
    logger.info("帕累托前沿 (目标值: [-T_M1, -T_M2, -T_M3]):")
    logger.info(f"\n{res.F}")
    logger.info("对应的分配方案 (决策变量):")
    logger.info(f"\n{res.X}")

    # 选择木桶短板最长的解作为最佳方案
    best_idx = np.argmax(np.min(-res.F, axis=1))
    best_solution = res.X[best_idx]
    best_objectives = -res.F[best_idx]
    
    logger.info("\n--- 最佳均衡策略 ---")
    logger.info(f"分配矩阵:\n{best_solution.reshape(5,3)}")
    logger.info(f"遮蔽时长: T_M1={best_objectives[0]:.4f}s, T_M2={best_objectives[1]:.4f}s, T_M3={best_objectives[2]:.4f}s")
    
    # 保存结果到文件
    np.savez(f"{P5Config.RESULTS_DIR}/final_results.npz", F=res.F, X=res.X, history=res.history)
    logger.info(f"最终结果已保存到 {P5.RESULTS_DIR}/final_results.npz")

    return {"problem_id": 5, "best_solution": best_solution, "best_objectives": best_objectives}