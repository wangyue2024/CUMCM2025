# solvers_p5/p5_solver.py

import os
import pickle
import time
import numpy as np
from multiprocessing import Pool

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization

from .p5_config import P5Config
from .p5_utils import setup_logger
from .p5_components import AssignmentProblem, AssignmentRepair, SmartInitialization
from .p5_parallel_evaluator import ParallelEvaluator

logger = setup_logger()

def solve_problem_5():
    """问题五的主求解函数"""
    logger.info("="*50)
    logger.info("开始求解问题五：多无人机多目标协同策略")
    logger.info("="*50)

    start_time = time.time()

    # 1. 初始化并行池和评估器
    # 注意：Pool对象不能被pickle，所以我们在这里创建并传递给pymoo
    pool = Pool(processes=P5Config.CPU_CORES)
    evaluator = ParallelEvaluator()
    problem = AssignmentProblem(evaluator)
    problem.parallelization = StarmapParallelization(pool.starmap)

    # 2. 检查是否有断点文件
    if os.path.exists(P5Config.CHECKPOINT_FILE):
        with open(P5Config.CHECKPOINT_FILE, "rb") as f:
            algorithm = pickle.load(f)
        logger.info(f"成功从 {P5Config.CHECKPOINT_FILE} 加载算法状态，继续运行。")
    else:
        algorithm = NSGA2(
            pop_size=P5Config.POP_SIZE,
            sampling=SmartInitialization(),
            # 使用简单的交叉和变异，配合修复算子
            crossover=SBX(prob=0.9, eta=15, repair=AssignmentRepair()),
            mutation=PM(eta=20, repair=AssignmentRepair()),
            eliminate_duplicates=True
        )
        logger.info("未找到快照文件，开始新的优化任务。")

    # 3. 运行优化
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', P5Config.N_GENERATIONS),
        seed=1,
        verbose=True,
        save_history=True,
        # 定期保存快照
        callback=lambda alg: pickle.dump(alg, open(P5Config.CHECKPOINT_FILE, "wb"))
    )

    pool.close()
    pool.join()
    end_time = time.time()
    logger.info(f"问题五求解完成，总耗时: {(end_time - start_time) / 3600:.2f} 小时")

    # 4. 保存并分析最终结果
    logger.info("帕累托前沿 (目标值: [T_M1, T_M2, T_M3]):")
    logger.info(f"\n{-res.F}") # 打印正时长
    
    # 选择木桶短板最长的解作为最佳推荐方案
    best_idx = np.argmax(np.min(-res.F, axis=1))
    best_solution_x = res.X[best_idx]
    best_objectives = -res.F[best_idx]
    
    logger.info("\n--- 最佳均衡策略 (推荐方案) ---")
    logger.info(f"分配方案 (15维向量): {best_solution_x.tolist()}")
    logger.info(f"遮蔽时长: T_M1={best_objectives[0]:.4f}s, T_M2={best_objectives[1]:.4f}s, T_M3={best_objectives[2]:.4f}s")
    
    # 保存结果
    np.savez(f"{P5Config.RESULTS_DIR}/final_results.npz", F=res.F, X=res.X, history=res.history)
    logger.info(f"最终结果已保存到 {P5Config.RESULTS_DIR}/final_results.npz")

    return res