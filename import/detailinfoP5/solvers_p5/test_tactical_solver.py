# test_tactical_solver.py

import time
import json
from solvers_p5.tactical_solver import solve_tactical_problem
import numpy as np
def run_tests():
    """运行一系列测试用例来验证战术求解器。"""
    print("="*80)
    print("开始测试问题五的下层战术求解器...")
    print("="*80)

    # 定义求解器精度
    solver_params = {'maxfevals': 10000, 'popsize': 100}

    # 定义测试用例
    test_cases = [
        {
            "name": "Case 1: FY1, 1弹打M1 (类似问题二)",
            "uav_id": "FY1",
            "task_list": [(0, 'M1')]
        },
        {
            "name": "Case 2: FY1, 3弹打M1 (类似问题三)",
            "uav_id": "FY1",
            "task_list": [(0, 'M1'), (1, 'M1'),(2,'M1')]
        },
        {
            "name": "Case 3: FY3, 2弹分别打M2, M3 (核心测试)",
            "uav_id": "FY3",
            "task_list": [(0, 'M1')]
        },
        {
            "name": "Case 4: FY4, 3弹打3个不同目标",
            "uav_id": "FY4",
            "task_list": [(0, 'M1'), (1, 'M2'), (2, 'M3')]
        },
        {
            "name": "Case 5: FY5, 无任务 (边缘测试)",
            "uav_id": "FY5",
            "task_list": []
        }
    ]

    for case in test_cases:
        print(f"\n--- Running Test: {case['name']} ---")
        
        start_time = time.perf_counter()
        
        results = solve_tactical_problem(
            uav_id=case['uav_id'],
            task_list=case['task_list'],
            solver_params=solver_params
        )
        
        end_time = time.perf_counter()
        
        print(f"求解耗时: {end_time - start_time:.2f} 秒")
        
        # 使用json进行格式化输出，非常适合观察复杂的嵌套字典
        print("返回结果详情:")
        print(json.dumps(results, indent=4, default=lambda x: str(x) if isinstance(x, np.ndarray) else x))
        
        # 简单验证
        if results['status'] == 'OPTIMIZED':
            contributions = results['objective_contributions']
            print(f"对各导弹的贡献: M1={contributions['M1']:.2f}s, M2={contributions['M2']:.2f}s, M3={contributions['M3']:.2f}s")
        else:
            print(f"状态: {results['status']}")

    print("\n" + "="*80)
    print("所有测试用例执行完毕。")
    print("="*80)

if __name__ == "__main__":
    run_tests()