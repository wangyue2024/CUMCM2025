import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 1. 数据定义 (与之前相同) ---
# 无人机和导弹的原始坐标，用于计算方向
uavs = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300])
}

missiles = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900])
}

# 假目标坐标
false_target_pos = np.array([0, 0, 0])

# 真目标信息
true_target_base_center = (0, 200, 0)
true_target_radius = 7
true_target_height = 10

# --- 2. 设置绘图环境 ---
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形和三维坐标系
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# --- 3. 绘制近景核心元素 ---

# 绘制假目标 (黑色)
ax.scatter(false_target_pos[0], false_target_pos[1], false_target_pos[2], c='black', marker='X', s=200, label='假目标')
ax.text(false_target_pos[0] + 10, false_target_pos[1] + 10, false_target_pos[2], '假目标 (0,0,0)', color='black', fontsize=12)

# 绘制真目标 (绿色圆柱体)
# 创建圆柱的网格数据
z = np.linspace(0, true_target_height, 50)
theta = np.linspace(0, 2 * np.pi, 50)
theta_grid, z_grid = np.meshgrid(theta, z)
# 计算圆柱表面坐标
x_grid = true_target_radius * np.cos(theta_grid) + true_target_base_center[0]
y_grid = true_target_radius * np.sin(theta_grid) + true_target_base_center[1]
# 绘制圆柱表面
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.6, color='green')
ax.text(true_target_base_center[0], true_target_base_center[1] + 20, 0, '真目标', color='green', fontsize=12)


# --- 4. 绘制表示方向的航向线段 ---

# 定义航向指示线的可见长度
line_segment_length = 400 

# 绘制导弹航向线段 (蓝色)
for name, pos in missiles.items():
    # 计算从导弹位置指向假目标(原点)的单位向量
    direction_vector = -pos / np.linalg.norm(pos)
    # 定义线段的起点和终点
    # 起点在远处，终点靠近目标
    start_point = false_target_pos - direction_vector * (line_segment_length + 50)
    end_point = false_target_pos - direction_vector * 50 # 在距离目标50米处结束，避免重叠
    ax.plot([start_point[0], end_point[0]], 
            [start_point[1], end_point[1]], 
            [start_point[2], end_point[2]], 
            'b--', lw=1.5)
    # 在航向线的起点标注来源
    ax.text(start_point[0], start_point[1], start_point[2], f'来自 {name}', color='blue')

# 绘制无人机航向线段 (红色)
for name, pos in uavs.items():
    # 计算从无人机位置指向假目标(原点)的单位向量
    direction_vector = -pos / np.linalg.norm(pos)
    # 定义线段的起点和终点
    start_point = false_target_pos - direction_vector * (line_segment_length + 50)
    end_point = false_target_pos - direction_vector * 50
    ax.plot([start_point[0], end_point[0]], 
            [start_point[1], end_point[1]], 
            [start_point[2], end_point[2]], 
            'r--', lw=1.5)
    # 在航向线的起点标注来源
    ax.text(start_point[0], start_point[1], start_point[2], f'来自 {name}', color='red')


# --- 5. 图表美化 ---
ax.set_xlabel('X 轴 (米)')
ax.set_ylabel('Y 轴 (米)')
ax.set_zlabel('Z 轴 (米)')
ax.set_title('目标区域近景态势图', fontsize=16)

# 设置新的、聚焦于目标区域的坐标轴范围
ax.set_xlim([-500, 500])
ax.set_ylim([-200, 600])
ax.set_zlim([0, 500])

# 设置合适的视角
ax.view_init(elev=30, azim=45)

# 创建图例
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

legend_elements = [
    Line2D([0], [0], marker='X', color='w', label='假目标', markerfacecolor='k', markersize=12),
    mpatches.Patch(color='green', alpha=0.6, label='真目标 (圆柱体)'),
    Line2D([0], [0], linestyle='--', color='b', lw=2, label='导弹来袭方向'),
    Line2D([0], [0], linestyle='--', color='r', lw=2, label='无人机接近方向')
]
ax.legend(handles=legend_elements, loc='upper right')

# 显示图形
plt.tight_layout()
plt.show()