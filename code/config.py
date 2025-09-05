# config.py
import numpy as np

# --- 物理常量 ---
G = 9.81  # 重力加速度 (m/s^2)

# --- 场景参数 ---
# 目标位置
P_FALSE_TARGET = np.array([0, 0, 0])
P_TRUE_TARGET = np.array([0, 200, 0])

# 导弹初始位置 (M1, M2, M3)
MISSILE_INITIAL_POS = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900])
}
MISSILE_VELOCITY = 300.0  # m/s

# 无人机初始位置 (FY1 to FY5)
UAV_INITIAL_POS = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300])
}
UAV_VELOCITY_RANGE = [70.0, 140.0]

# 烟幕参数
SMOKE_EFFECTIVE_RADIUS = 10.0  # m
SMOKE_EFFECTIVE_DURATION = 20.0  # s
SMOKE_SINK_VELOCITY = 3.0 # m/s