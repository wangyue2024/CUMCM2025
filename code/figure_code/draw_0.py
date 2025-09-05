import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 1. 数据定义 ---
# 根据题目信息，定义无人机和导弹的坐标
uavs = {
    'FY1': (17800, 0, 1800),
    'FY2': (12000, 1400, 1400),
    'FY3': (6000, -3000, 700),
    'FY4': (11000, 2000, 1800),
    'FY5': (13000, -2000, 1300)
}

missiles = {
    'M1': (20000, 0, 2000),
    'M2': (19000, 600, 2100),
    'M3': (18000, -600, 1900)
}

# 假目标坐标
false_target_pos = (0, 0, 0)

# 真目标信息
true_target_base_center = (0, 200, 0)
true_target_radius = 7
true_target_height = 10

# --- 2. 设置绘图环境 ---
# 设置支持中文的字体，以防标签显示为方框
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 创建一个图形和三维坐标系
fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# --- 3. 绘制各个元素 ---

# 绘制无人机 (红色)
for name, pos in uavs.items():
    ax.scatter(pos[0], pos[1], pos[2], c='red', marker='o', s=50, label=f'无人机 {name}' if name == 'FY1' else "")
    ax.text(pos[0], pos[1], pos[2] + 100, f' {name}', color='red')

# 绘制导弹 (蓝色)
for name, pos in missiles.items():
    ax.scatter(pos[0], pos[1], pos[2], c='blue', marker='^', s=50, label=f'导弹 {name}' if name == 'M1' else "")
    ax.text(pos[0], pos[1], pos[2] + 100, f' {name}', color='blue')

# 绘制假目标 (黑色)
ax.scatter(false_target_pos[0], false_target_pos[1], false_target_pos[2], c='black', marker='X', s=150, label='假目标')
ax.text(false_target_pos[0] + 500, false_target_pos[1], false_target_pos[2], '假目标 (0,0,0)', color='black')

# 绘制真目标 (绿色圆柱体)
# 创建圆柱的网格数据
z = np.linspace(0, true_target_height, 50)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, z)
# 计算圆柱表面坐标
x_grid = true_target_radius * np.cos(theta_grid) + true_target_base_center[0]
y_grid = true_target_radius * np.sin(theta_grid) + true_target_base_center[1]
# 绘制圆柱表面
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='green', label='真目标')
# 为图例创建一个代理对象
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='无人机 (UAVs)', markerfacecolor='r', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='导弹 (Missiles)', markerfacecolor='b', markersize=10),
    Line2D([0], [0], marker='X', color='w', label='假目标', markerfacecolor='k', markersize=10),
    plt.Rectangle((0,0),1,1,fc="green", alpha=0.5, label='真目标'),
    Line2D([0], [0], linestyle='--', color='b', label='导弹航向'),
    Line2D([0], [0], linestyle='--', color='r', label='无人机航向 (假设)')
]


# --- 4. 绘制连线 ---

# 绘制导弹到假目标的虚线 (蓝色)
for name, pos in missiles.items():
    ax.plot([pos[0], false_target_pos[0]], 
            [pos[1], false_target_pos[1]], 
            [pos[2], false_target_pos[2]], 
            'b--')

# 绘制无人机朝向假目标的虚线 (红色)
for name, pos in uavs.items():
    ax.plot([pos[0], false_target_pos[0]], 
            [pos[1], false_target_pos[1]], 
            [pos[2], false_target_pos[2]], 
            'r--')

# --- 5. 图表美化 ---
ax.set_xlabel('X 轴 (米)')
ax.set_ylabel('Y 轴 (米)')
ax.set_zlabel('Z 轴 (米)')
ax.set_title('基本情况示意图', fontsize=16)

# 设置坐标轴范围以获得更好的视图
ax.set_xlim([0, 22000])
ax.set_ylim([-4000, 4000])
ax.set_zlim([0, 2500])

# 设置初始视角
ax.view_init(elev=25, azim=-75)

# 添加图例
ax.legend(handles=legend_elements, loc='upper left')

# 显示图形
plt.show()