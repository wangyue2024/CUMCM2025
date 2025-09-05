# optimizers/problem_solvers.py

import cma
import numpy as np
from model.physics_model import PhysicsModelWithCylinder
import config

def solve_problem_2():
    print("--- 开始求解问题2 ---")
    
    # 1. 初始化配置和模型
    cfg = config.Config()
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)
    
    # 2. 设置CMA-ES参数
    # 决策变量: [速度, 方向角, 投放时间, 起爆延迟]
    initial_guess = [100, np.pi/2, 5.0, 5.0] # 初始猜测
    sigma0 = 3.0 # 初始步长
    
    # 边界条件
    bounds = [
        [cfg.V_UAV_MIN, 0, 0, 1], # 下界
        [cfg.V_UAV_MAX, 2*np.pi, 30, 10] # 上界 (投放和延迟时间可根据情况调整)
    ]
    
    # 3. 运行优化器
    # cma.fmin2的第一个参数直接就是我们的成本函数
    best_solution, es = cma.fmin2(
        model.cost_function_q2,
        initial_guess,
        sigma0,
        options={'bounds': bounds, 'maxfevals': 2000} # 增加评估次数
    )
    
    # 4. 分析和输出结果
    print("\n--- 优化完成 ---")
    final_cost = es.result.fbest
    print(f"最优成本值: {final_cost:.4f}")
    
    # 重新计算最优策略的详细指标
    shielding_time, miss_dist, details = model.calculate_shielding_metrics(
        best_solution[0], best_solution[1], [best_solution[2]], [best_solution[3]]
    )
    
    print(f"最大有效遮蔽时长: {shielding_time:.4f} s")
    print(f"最优策略 [速度, 角度, 投放时间, 延迟]:")
    print(f"  速度: {best_solution[0]:.2f} m/s")
    print(f"  角度: {np.rad2deg(best_solution[1]):.2f} 度")
    print(f"  投放时间: {best_solution[2]:.2f} s")
    print(f"  起爆延迟: {best_solution[3]:.2f} s")
    
    # ... 在这里可以调用可视化函数，使用details字典绘图 ...

if __name__ == '__main__':
    solve_problem_2()