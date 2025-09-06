# solvers_p5/p5_parallel_evaluator.py

import numpy as np
from .p5_config import P5Config
from .p5_utils import setup_logger
from solvers_p5.tactical_solver import solve_tactical_problem

logger = setup_logger()

class ParallelEvaluator:
    def __init__(self):
        self.missile_ids = ['M1', 'M2', 'M3']

    def _get_tactical_solver_params(self, algorithm):
        """根据当前代数返回对应的下层求解器精度"""
        current_gen = algorithm.n_gen if algorithm and hasattr(algorithm, 'n_gen') else 0
        total_gens = P5Config.N_GENERATIONS
        
        for stage_ratio, params in P5Config.TACTICAL_SOLVER_STAGES:
            if current_gen / total_gens <= stage_ratio:
                return params
        return P5Config.TACTICAL_SOLVER_STAGES[-1][1]

    def evaluate(self, x, algorithm):
        """
        评估单个分配方案x。
        这是在并行worker中执行的函数。
        """
        solver_params = self._get_tactical_solver_params(algorithm)
        
        # 1. 解析任务分配
        tasks_per_uav = []
        for i in range(5):
            task_list = []
            smokes = x[i*3 : i*3+3]
            for smoke_idx, target_missile_num in enumerate(smokes):
                if target_missile_num > 0:
                    missile_id = f"M{target_missile_num}"
                    task_list.append((smoke_idx, missile_id))
            tasks_per_uav.append(task_list)

        # 2. 串行计算每个无人机的贡献 (因为这是在单个worker内部)
        all_smoke_details = []
        for i in range(5):
            uav_id = f"FY{i+1}"
            task_list = tasks_per_uav[i]
            # 调用下层战术求解器
            result = solve_tactical_problem(uav_id, task_list, solver_params)
            if result and result.get("smoke_details"):
                all_smoke_details.extend(result["smoke_details"])

        # 3. 聚合结果
        intervals_per_missile = {'M1': [], 'M2': [], 'M3': []}
        for detail in all_smoke_details:
            missile_id = detail['target_missile_id']
            intervals_per_missile[missile_id].extend(detail['shielding_intervals'])
        
        final_objectives = []
        for missile_id in self.missile_ids:
            intervals = intervals_per_missile[missile_id]
            if not intervals:
                total_time = 0.0
            else:
                intervals.sort(key=lambda item: item[0])
                merged = [intervals[0]]
                for start, end in intervals[1:]:
                    if start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                total_time = sum(end - start for start, end in merged)
            final_objectives.append(total_time)
            
        # 4. 日志记录
        gen = algorithm.n_gen if algorithm else 'N/A'
        obj_str = ", ".join([f"{obj:.2f}s" for obj in final_objectives])
        logger.info(f"[Gen {gen}] Evaluated individual {x.tolist()}. Objectives (T_M1, T_M2, T_M3): [{obj_str}]")

        return np.array(final_objectives)