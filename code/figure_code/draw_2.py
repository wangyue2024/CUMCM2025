import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# Part 1: Recreate necessary Config and Helper Functions
# ==============================================================================

class Config:
    """
    A class to hold all configuration parameters, based on your provided scripts.
    """
    G = 9.8
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    CYLINDER_CENTER_XY = np.array([0.0, 200.0])
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0
    CYLINDER_SAMPLE_POINTS_EDGE = 10 # Number of points to sample on edges
    V_MISSILE = 300.0
    MISSILE_INITIAL_POS = {'M1': np.array([20000.0, 0.0, 2000.0])}
    UAV_INITIAL_POS = {'FY1': np.array([17800.0, 0.0, 1800.0])}
    R_SMOKE = 10.0

def generate_cylinder_sample_points(cfg):
    """Generates sample points on the surface of the true target cylinder."""
    points = []
    cx, cy = cfg.CYLINDER_CENTER_XY
    radius = cfg.CYLINDER_RADIUS
    height = cfg.CYLINDER_HEIGHT
    num_edge_points = cfg.CYLINDER_SAMPLE_POINTS_EDGE
    
    points.append(np.array([cx, cy, 0]))
    points.append(np.array([cx, cy, height]))
    
    for i in range(num_edge_points):
        angle = 2 * np.pi * i / num_edge_points
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        
        points.append(np.array([x, y, 0]))
        points.append(np.array([x, y, height]))
        points.append(np.array([x, y, height / 2.0]))
        
    return np.array(points)

# ==============================================================================
# Part 2: Main execution block for calculation and visualization
# ==============================================================================

if __name__ == '__main__':
    # --- 1. Initialize Configuration ---
    cfg = Config()
    
    # --- 2. Use the ACTUAL Optimal Strategy from your CMA-ES run ---
    optimal_strategy = {
        'speed': 97.50,                # m/s
        'theta_rad': np.deg2rad(178.09), # Convert degrees to radians
        't_launch': 0.00,              # s
        'dt_det': 2.87                 # s
    }
    t_detonation = optimal_strategy['t_launch'] + optimal_strategy['dt_det']

    # --- 3. Calculate Positions based on the Optimal Strategy ---
    
    # UAV calculations
    p_uav_0 = cfg.UAV_INITIAL_POS['FY1']
    v_vec_uav = np.array([
        optimal_strategy['speed'] * np.cos(optimal_strategy['theta_rad']),
        optimal_strategy['speed'] * np.sin(optimal_strategy['theta_rad']),
        0
    ])
    p_uav_at_detonation = p_uav_0 + v_vec_uav * t_detonation

    # Missile calculations
    p_missile_0 = cfg.MISSILE_INITIAL_POS['M1']
    direction_vec_m = cfg.P_FALSE_TARGET - p_missile_0
    u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
    v_vec_missile = cfg.V_MISSILE * u_missile
    p_missile_at_detonation = p_missile_0 + v_vec_missile * t_detonation

    # Smoke grenade detonation position
    # Since t_launch is 0, p_launch is the same as p_uav_0
    p_launch = p_uav_0 + v_vec_uav * optimal_strategy['t_launch']
    displacement_fall = np.array([
        v_vec_uav[0] * optimal_strategy['dt_det'],
        v_vec_uav[1] * optimal_strategy['dt_det'],
        -0.5 * cfg.G * optimal_strategy['dt_det']**2
    ])
    p_detonation = p_launch + displacement_fall

    # Generate target sample points
    target_sample_points = generate_cylinder_sample_points(cfg)

    # --- 4. Setup the 3D Plot ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- 5. Plot all elements ---

    # Plot UAV at detonation time (Red dot)
    ax.scatter(p_uav_at_detonation[0], p_uav_at_detonation[1], p_uav_at_detonation[2], 
               c='red', marker='o', s=80)
    ax.text(p_uav_at_detonation[0], p_uav_at_detonation[1], p_uav_at_detonation[2] + 100, 
            f'FY1\n({p_uav_at_detonation[0]:.0f}, {p_uav_at_detonation[1]:.0f}, {p_uav_at_detonation[2]:.0f})', color='red')

    # Plot Missile at detonation time (Blue dot)
    ax.scatter(p_missile_at_detonation[0], p_missile_at_detonation[1], p_missile_at_detonation[2], 
               c='blue', marker='^', s=80)
    ax.text(p_missile_at_detonation[0], p_missile_at_detonation[1], p_missile_at_detonation[2] + 100, 
            f'M1\n({p_missile_at_detonation[0]:.0f}, {p_missile_at_detonation[1]:.0f}, {p_missile_at_detonation[2]:.0f})', color='blue')

    # Plot True Target (Green Cylinder)
    z_cyl = np.linspace(0, cfg.CYLINDER_HEIGHT, 50)
    theta_cyl = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
    x_grid = cfg.CYLINDER_RADIUS * np.cos(theta_grid) + cfg.CYLINDER_CENTER_XY[0]
    y_grid = cfg.CYLINDER_RADIUS * np.sin(theta_grid) + cfg.CYLINDER_CENTER_XY[1]
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.4, color='green')

    # Plot the sample points on the cylinder (Yellow dots)
    ax.scatter(target_sample_points[:, 0], target_sample_points[:, 1], target_sample_points[:, 2],
               c='yellow', marker='o', s=50, edgecolors='black', depthshade=False, zorder=10)

    # Plot False Target (Black 'X')
    ax.scatter(cfg.P_FALSE_TARGET[0], cfg.P_FALSE_TARGET[1], cfg.P_FALSE_TARGET[2], 
               c='black', marker='X', s=150)
    ax.text(cfg.P_FALSE_TARGET[0] + 500, cfg.P_FALSE_TARGET[1], cfg.P_FALSE_TARGET[2], 
            '假目标 (0,0,0)', color='black')

    # Plot Missile Trajectory (Blue dashed line)
    ax.plot([p_missile_0[0], cfg.P_FALSE_TARGET[0]], [p_missile_0[1], cfg.P_FALSE_TARGET[1]], [p_missile_0[2], cfg.P_FALSE_TARGET[2]], 'b--')

    # Plot UAV Trajectory (Red dashed line)
    ax.plot([p_uav_0[0], p_uav_at_detonation[0]], [p_uav_0[1], p_uav_at_detonation[1]], [p_uav_0[2], p_uav_at_detonation[2]], 'r--')

    # Plot the Smoke Cloud (Gray sphere)
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_smoke = cfg.R_SMOKE * np.cos(u) * np.sin(v) + p_detonation[0]
    y_smoke = cfg.R_SMOKE * np.sin(u) * np.sin(v) + p_detonation[1]
    z_smoke = cfg.R_SMOKE * np.cos(v) + p_detonation[2]
    ax.plot_surface(x_smoke, y_smoke, z_smoke, color='gray', alpha=0.4)
    ax.text(p_detonation[0], p_detonation[1], p_detonation[2], '烟幕', color='dimgray')

    # --- 6. Finalize and Show Plot ---
    ax.set_xlabel('X 轴 (米)')
    ax.set_ylabel('Y 轴 (米)')
    ax.set_zlabel('Z 轴 (米)')
    ax.set_title(f'问题二最优策略可视化 (t={t_detonation:.2f}s 爆炸瞬间)', fontsize=16)
    
    ax.set_xlim([0, 21000])
    ax.set_ylim([-1500, 1500])
    ax.set_zlim([0, 2500])
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='无人机 (爆炸时刻)', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='导弹 (爆炸时刻)', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='假目标', markerfacecolor='k', markersize=10),
        mpatches.Patch(color='green', alpha=0.4, label='真目标'),
        Line2D([0], [0], marker='o', color='w', label='真目标采样点', markerfacecolor='yellow', markeredgecolor='k', markersize=8),
        mpatches.Patch(color='gray', alpha=0.4, label='烟幕云'),
        Line2D([0], [0], linestyle='--', color='r', label='无人机轨迹'),
        Line2D([0], [0], linestyle='--', color='b', label='导弹轨迹')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.view_init(elev=20, azim=-75)
    plt.tight_layout()
    plt.show()