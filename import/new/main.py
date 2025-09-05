# main.py (修改版)

import time
import numpy as np

try:
    import cma
except ImportError:
    print("错误: 未找到 'cma' 库。请运行 'pip install cma' 进行安装。")
    exit()

# 导入新的高级求解器
from problem_solvers import solve_problem_1, solve_problem_2_advanced, solve_problem_3_advanced

def main():
    np.set_printoptions(precision=4, suppress=True)
    
    print("="*60)
    print("开始执行：烟幕干扰弹的投放策略问题求解 (高级优化版)")
    print("="*60)

    # --- 问题 1 ---
    start_time = time.time()
    solve_problem_1()
    end_time = time.time()
    print(f"问题 1 求解耗时: {end_time - start_time:.4f} 秒")

    # --- 问题 2 (调用新版) ---
    start_time = time.time()
    solve_problem_2_advanced()
    end_time = time.time()
    print(f"问题 2 求解耗时: {end_time - start_time:.2f} 秒")

    # --- 问题 3 (调用新版) ---
    start_time = time.time()
    solve_problem_3_advanced()
    end_time = time.time()
    print(f"问题 3 求解耗时: {end_time - start_time:.2f} 秒")

    print("\n" + "="*60)
    print("所有任务执行完毕。")
    print("="*60)

if __name__ == "__main__":
    main()