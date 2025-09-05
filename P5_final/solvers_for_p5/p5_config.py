# solvers_for_p5/p5_config.py

class P5Config:
    # --- 并行与性能控制 ---
    # 使用的CPU核心数，0表示使用所有可用核心
    CPU_CORES = 0 
    
    # --- 上层遗传算法 (NSGA-II) 参数 ---
    POP_SIZE = 80          # 种群大小 (推荐4的倍数)
    N_GENERATIONS = 200    # 总迭代代数
    
    # --- 下层战术优化器 (CMA-ES) 精度控制 ---
    # 定义不同阶段的精度参数 [('阶段比例', {'maxfevals': fevals, 'popsize': ps}), ...]
    TACTICAL_SOLVER_STAGES = [
        (0.3, {'maxfevals': 5000, 'popsize': 30}),   # 前30%的代数：快速评估
        (0.7, {'maxfevals': 15000, 'popsize': 40}),  # 30%-70%的代数：中等精度
        (1.0, {'maxfevals': 30000, 'popsize': 50})   # 最后阶段：全力优化
    ]

    # --- 结果与日志 ---
    RESULTS_DIR = "results/problem5"  # 结果保存目录
    LOG_FILE = f"{RESULTS_DIR}/p5_optimization.log" # 日志文件
    CHECKPOINT_FILE = f"{RESULTS_DIR}/checkpoint.pkl" # 断点续跑文件