# test_p5_cost_function.py

import numpy as np
import cma
import time
from typing import Dict

# 确保可以从您的项目结构中导入这些模块
from config import Config
from models.physics_model_analytical import PhysicsModelAnalytical

def print_detailed_results(details: Dict):
    """
    以一种美观、详细的格式打印评估函数的返回结果。
    """
    print("\n" + "="*40 + " 详细评估结果 " + "="*40)
    
    # 1. 打印总体目标贡献
    print("\n[+] 对各导弹的最终遮蔽时长贡献:")
    contributions = details['objective_contributions']
    for missile_id, duration in contributions.items():
        print(f"  - {missile_id}: {duration:.4f} 秒")
    
    total_time = sum(contributions.values())
    print(f"  - (简单求和，非合并): {total_time:.4f} 秒")
    
    # 2. 打印标量成本
    print(f"\n[+] 用于CMA-ES的标量成本值: {details['scalar_cost']:.6f}")

    # 3. 打印每个烟幕弹的详细信息
    print("\n[+] 各烟幕弹详情:")
    header = (
        f"{'弹药索引':>8s} | {'目标导弹':>10s} | {'投放时间(s)':>12s} | {'爆炸时间(s)':>12s} | "
        f"{'独立时长(s)':>12s} | {'代理成本':>12s}"
    )
    print(header)
    print("-" * len(header))

    for smoke in details['smoke_details']:
        row = (
            f"{smoke['smoke_index']:>8d} | {smoke['target_missile_id']:>10s} | {smoke['launch_time']:>12.4f} | "
            f"{smoke['detonation_time']:>12.4f} | {smoke['individual_shielding_time']:>12.4f} | "
            f"{smoke['proxy_cost']:>12.2e}"
        )
        print(row)
        
        lp = smoke['launch_point']
        dp = smoke['detonation_point']
        print(f"  └─ 投放点: ({lp[0]:.1f}, {lp[1]:.1f}, {lp[2]:.1f}) | "
              f"爆炸点: ({dp[0]:.1f}, {dp[1]:.1f}, {dp[2]:.1f})")
        
        intervals_str = ", ".join([f"[{start:.2f}s, {end:.2f}s]" for start, end in smoke['shielding_intervals']])
        if not intervals_str:
            intervals_str = "无"
        print(f"  └─ 遮蔽区间: {intervals_str}")

    print("="*95)


def main():
    """测试脚本主函数"""
    print("--- 开始测试：单无人机-多目标成本函数 ---")

    # --- 1. 定义测试场景 ---
    uav_id_to_test = 'FY1'
    # 任务: 弹0打M1, 弹1打M2, 弹2打M1
    task_list_to_test = [(0, 'M1'), (1, 'M2'), (2, 'M1')]
    print(f"测试无人机: {uav_id_to_test}")
    print(f"测试任务列表: {task_list_to_test}")

    # --- 2. 初始化模型 ---
    cfg = Config()
    # 注意：虽然模型初始化需要一个missile_id，但我们的新函数会在内部动态切换，
    # 所以这里的初始ID对于本次测试并不关键。
    model = PhysicsModelAnalytical(missile_id='M1', config_obj=cfg)

    # --- 3. 设置CMA-ES优化器 ---
    # 决策向量x: [v, theta, t_l0, dt_d0, t_l1, dt_d1, t_l2, dt_d2] (8维)
    initial_guess = [
        100, np.pi / 4,  # v, theta
        1.0, 1.0,        # 弹0: t_launch, dt_det
        2.0, 2.0,       # 弹1: t_launch, dt_det
        3.0, 3.0        # 弹2: t_launch, dt_det
    ]
    sigma0 = 5.0
    
    # 直接在物理空间优化，设置边界
    bounds_lower = [cfg.V_UAV_MIN, 0, 0, 0, 0, 0, 0, 0]
    bounds_upper = [cfg.V_UAV_MAX, 2*np.pi, 20, 20, 20, 20, 20, 20]

    options = {
        'bounds': [bounds_lower, bounds_upper],
        'maxfevals': 5000, # 短期运行，仅用于测试
        'popsize': 300,
        'verbose': -9,
        'seed': int(time.time())
    }

    # 使用lambda表达式将额外参数绑定到成本函数上
    objective_function = lambda x: model.cost_function_p5_scalar(
        x,
        uav_id=uav_id_to_test,
        task_list=task_list_to_test
    )

    es = cma.CMAEvolutionStrategy(initial_guess, sigma0, options)
    
    print("\n--- 开始模拟优化过程 (短期运行) ---")
    
    # --- 4. 运行模拟优化循环 ---
    iteration = 0
    while not es.stop():
        solutions = es.ask()
        costs = [objective_function(s) for s in solutions]
        es.tell(solutions, costs)
        
        if iteration % 5 == 0:
            print(f"迭代: {iteration:3d}, "
                  f"评估次数: {es.countevals:4d}, "
                  f"当前最优成本: {es.result.fbest:.6f}")
        iteration += 1
    
    print("--- 模拟优化结束 ---")

    # --- 5. 对找到的“最优”解进行详细评估和打印 ---
    best_solution_phys = es.result.xbest
    
    print("\n对模拟找到的最优解进行最终详细评估...")
    final_details = model.evaluate_single_uav_multi_target(
        uav_id=uav_id_to_test,
        x_phys=best_solution_phys,
        task_list=task_list_to_test
    )

    print_detailed_results(final_details)
    
    print("\n测试脚本执行完毕。如果无错误且输出内容合理，则函数基本功能正常。")


if __name__ == "__main__":
    main()