# solvers_p5/p5_config.py
class P5Config:
    CPU_CORES = 30 # 在32核服务器上留一些余量
    POP_SIZE = 80
    N_GENERATIONS = 200
    TACTICAL_SOLVER_STAGES = [
        (0.3, {'maxfevals': 5000, 'popsize': 30}),
        (0.7, {'maxfevals': 15000, 'popsize': 40}),
        (1.0, {'maxfevals': 30000, 'popsize': 50})
    ]
    RESULTS_DIR = "results/problem5"
    LOG_FILE = f"{RESULTS_DIR}/p5_optimization.log"
    CHECKPOINT_FILE = f"{RESULTS_DIR}/checkpoint.pkl"