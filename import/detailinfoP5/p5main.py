# main_p5.py

from solvers_p5.p5_solver import solve_problem_5

if __name__ == "__main__":
    # 确保在多进程环境中，主逻辑只被执行一次
    solve_problem_5()