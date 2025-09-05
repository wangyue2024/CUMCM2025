# main.py
import os
from optimizers import problem_solvers
# 假设你有一个 visualization.py 文件
# from utils import visualization 

def setup_directories():
    """创建结果和图形目录（如果不存在）"""
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/figures'):
        os.makedirs('results/figures')

if __name__ == "__main__":
    setup_directories()
    
    # --- 求解问题 3 ---
    # 运行优化器并获取结果
    p3_details, p3_optimizer_instance = problem_solvers.solve_problem_3()
    
    # --- 调用可视化 ---
    # 假设你已经编写了可视化函数
    print("\nGenerating visualizations for Problem 3...")
    # visualization.plot_3d_trajectory(p3_details, save_path='results/figures/problem3_trajectory.png')
    # visualization.plot_gantt_chart(p3_details, save_path='results/figures/problem3_gantt.png')
    # visualization.plot_convergence(p3_optimizer_instance, save_path='results/figures/problem3_convergence.png')
    print("Visualizations would be saved in results/figures/ (if implemented).")
    
    # --- 可以在这里继续调用其他问题的求解器 ---
    # print("\n--- Solving Problem 4 ---")
    # p4_details, p4_optimizer_instance = problem_solvers.solve_problem_4()
    # ...