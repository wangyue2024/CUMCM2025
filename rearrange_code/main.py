# main.py

import time
from optimizers import problem_solvers
from utils import visualization

def main():
    """
    数学建模竞赛项目主入口。
    按顺序调用各个问题的求解器，并处理结果。
    """
    print("="*60)
    print("开始执行：烟幕干扰弹的投放策略问题求解")
    print("="*60)

    # --- 问题 1 ---
    start_time = time.time()
    results_p1 = problem_solvers.solve_problem_1()
    end_time = time.time()
    print(f"问题 1 求解耗时: {end_time - start_time:.2f} 秒")
    # 可视化问题1的结果
    # visualization.plot_3d_scenario(results_p1['details'], "问题一：固定策略场景")

    # --- 问题 2 ---
    start_time = time.time()
    results_p2 = problem_solvers.solve_problem_2()
    end_time = time.time()
    print(f"问题 2 求解耗时: {end_time - start_time:.2f} 秒")
    # 可视化问题2的结果
    # visualization.plot_3d_scenario(results_p2['details'], "问题二：最优策略场景")
    # visualization.plot_convergence(results_p2['log'], "问题二：优化收敛曲线")

    # --- 问题 3 ---
    start_time = time.time()
    results_p3 = problem_solvers.solve_problem_3()
    end_time = time.time()
    print(f"问题 3 求解耗时: {end_time - start_time:.2f} 秒")
    # 可视化问题3的结果
    # visualization.plot_3d_scenario(results_p3['details'], "问题三：多弹药最优策略场景")
    # visualization.plot_convergence(results_p3['log'], "问题三：优化收敛曲线")
    # visualization.save_results_to_excel(results_p3, "results/result1.xlsx")

    # --- 问题 4 (待实现) ---
    # results_p4 = problem_solvers.solve_problem_4()
    # visualization.save_results_to_excel(results_p4, "results/result2.xlsx")

    # --- 问题 5 (待实现) ---
    # results_p5 = problem_solvers.solve_problem_5()
    # visualization.save_results_to_excel(results_p5, "results/result3.xlsx")

    print("\n" + "="*60)
    print("所有任务执行完毕。")
    print("="*60)

if __name__ == "__main__":
    main()