# config.py

import numpy as np

class Config:
    # --- 物理常量 ---
    G = 9.8  # 重力加速度 (m/s^2)

    # --- 目标信息 ---
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    # 为解析解简化，我们将真目标视为一个点（圆柱中心）
    # 这是一个必要的、合理的简化，以启用强大的解析方法
    P_TRUE_TARGET = np.array([0.0, 200.0, 5.0]) # 圆柱中心点 (z=height/2)

    # --- 导弹信息 ---
    V_MISSILE = 300.0  # 导弹速度 (m/s)
    MISSILE_INITIAL_POS = {
        'M1': np.array([20000.0, 0.0, 2000.0]),
        'M2': np.array([19000.0, 600.0, 2100.0]),
        'M3': np.array([18000.0, -600.0, 1900.0])
    }

    # --- 无人机信息 ---
    V_UAV_MIN = 70.0   # 无人机最小速度 (m/s)
    V_UAV_MAX = 140.0  # 无人机最大速度 (m/s)
    UAV_INITIAL_POS = {
        'FY1': np.array([17800.0, 0.0, 1800.0]),
        'FY2': np.array([12000.0, 1400.0, 1400.0]),
        'FY3': np.array([6000.0, -3000.0, 700.0]),
        'FY4': np.array([11000.0, 2000.0, 1800.0]),
        'FY5': np.array([13000.0, -2000.0, 1300.0])
    }

    # --- 烟幕弹信息 ---
    R_SMOKE = 10.0            # 烟幕有效半径 (m)
    T_SMOKE_EFFECTIVE = 20.0  # 烟幕有效持续时间 (s)
    V_SMOKE_SINK = np.array([0.0, 0.0, -3.0]) # 烟幕下沉速度矢量 (m/s)
    MIN_LAUNCH_INTERVAL = 1.0 # 最小投放间隔 (s)

    # --- 优化控制参数 ---
    # 新增：用于奖励塑造的惩罚因子，引导未产生遮蔽的烟幕
    REWARD_SHAPING_PENALTY_FACTOR = 1e-7