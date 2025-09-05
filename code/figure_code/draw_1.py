import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# Part 1: Reuse the provided Config class
# ==============================================================================

class Config:
    G = 9.8
    P_FALSE_TARGET = np.array([0.0, 0.0, 0.0])
    CYLINDER_CENTER_XY = np.array([0.0, 200.0])
    CYLINDER_RADIUS = 7.0
    CYLINDER_HEIGHT = 10.0
    V_MISSILE = 300.0
    MISSILE_INITIAL_POS = {'M1': np.array([20000.0, 0.0, 2000.0])}
    UAV_INITIAL_POS = {'FY1': np.array([17800.0, 0.0, 1800.0])}
    R_SMOKE = 10.0

# ==============================================================================
# Part 2: Main execution block for calculation and visualization
# ==============================================================================

if __name__ == '__main__':
    # --- 1. Initialize Configuration and Scenario Parameters ---
    cfg = Config()
    
    # Scenario parameters from Problem 1
    uav_speed = 120.0
    t_launch = 1.5  # s
    dt_det = 3.6    # s
    t_detonation = t_launch + dt_det

    # --- 2. Calculate Positions at the Moment of Detonation (t = 5.1s) ---
    
    # UAV calculations
    p_uav_0 = cfg.UAV_INITIAL_POS['FY1']
    direction_vec_uav = cfg.P_FALSE_TARGET - p_uav_0
    u_uav = direction_vec_uav / np.linalg.norm(direction_vec_uav)
    v_vec_uav = uav_speed * u_uav
    # UAV position at the moment of detonation
    p_uav_at_detonation = p_uav_0 + v_vec_uav * t_detonation

    # Missile calculations
    p_missile_0 = cfg.MISSILE_INITIAL_POS['M1']
    direction_vec_m = cfg.P_FALSE_TARGET - p_missile_0
    u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
    v_vec_missile = cfg.V_MISSILE * u_missile
    # Missile position at the moment of detonation
    p_missile_at_detonation = p_missile_0 + v_vec_missile * t_detonation

    # Smoke grenade detonation position calculation
    # Position where the grenade was launched
    p_launch = p_uav_0 + v_vec_uav * t_launch
    # Displacement of the grenade during its fall (inertial + gravitational)
    dx = v_vec_uav[0] * dt_det
    dy = v_vec_uav[1] * dt_det
    dz = -0.5 * cfg.G * dt_det**2
    # Final detonation position
    p_detonation = p_launch + np.array([dx, dy, dz])

    # --- 3. Setup the 3D Plot ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- 4. Plot all elements ---

    # 2. Plot UAV at detonation time (Red dot)
    ax.scatter(p_uav_at_detonation[0], p_uav_at_detonation[1], p_uav_at_detonation[2], 
               c='red', marker='o', s=80, label='无人机 (爆炸时刻)')
    ax.text(p_uav_at_detonation[0], p_uav_at_detonation[1], p_uav_at_detonation[2] + 100, 
            f'FY1\n({p_uav_at_detonation[0]:.0f}, {p_uav_at_detonation[1]:.0f}, {p_uav_at_detonation[2]:.0f})', color='red')

    # 3. Plot Missile at detonation time (Blue dot)
    ax.scatter(p_missile_at_detonation[0], p_missile_at_detonation[1], p_missile_at_detonation[2], 
               c='blue', marker='^', s=80, label='导弹 (爆炸时刻)')
    ax.text(p_missile_at_detonation[0], p_missile_at_detonation[1], p_missile_at_detonation[2] + 100, 
            f'M1\n({p_missile_at_detonation[0]:.0f}, {p_missile_at_detonation[1]:.0f}, {p_missile_at_detonation[2]:.0f})', color='blue')

    # 4. Plot True Target (Green Cylinder)
    z = np.linspace(0, cfg.CYLINDER_HEIGHT, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = cfg.CYLINDER_RADIUS * np.cos(theta_grid) + cfg.CYLINDER_CENTER_XY[0]
    y_grid = cfg.CYLINDER_RADIUS * np.sin(theta_grid) + cfg.CYLINDER_CENTER_XY[1]
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='green')

    # 5. Plot False Target (Black 'X')
    ax.scatter(cfg.P_FALSE_TARGET[0], cfg.P_FALSE_TARGET[1], cfg.P_FALSE_TARGET[2], 
               c='black', marker='X', s=150, label='假目标')
    ax.text(cfg.P_FALSE_TARGET[0] + 500, cfg.P_FALSE_TARGET[1], cfg.P_FALSE_TARGET[2], 
            '假目标 (0,0,0)', color='black')

    # 6. Plot Missile Trajectory (Blue dashed line)
    ax.plot([p_missile_0[0], cfg.P_FALSE_TARGET[0]], 
            [p_missile_0[1], cfg.P_FALSE_TARGET[1]], 
            [p_missile_0[2], cfg.P_FALSE_TARGET[2]], 
            'b--', label='导弹轨迹')

    # 7. Plot UAV Trajectory up to detonation point (Red dashed line)
    ax.plot([p_uav_0[0], p_uav_at_detonation[0]], 
            [p_uav_0[1], p_uav_at_detonation[1]], 
            [p_uav_0[2], p_uav_at_detonation[2]], 
            'r--', label='无人机轨迹')

    # 8. Plot the Smoke Cloud (Gray sphere)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = cfg.R_SMOKE * np.outer(np.cos(u), np.sin(v)) + p_detonation[0]
    y = cfg.R_SMOKE * np.outer(np.sin(u), np.sin(v)) + p_detonation[1]
    z = cfg.R_SMOKE * np.outer(np.ones(np.size(u)), np.cos(v)) + p_detonation[2]
    ax.plot_surface(x, y, z, color='gray', alpha=0.4, label='烟幕云 (爆炸瞬间)')
    ax.text(p_detonation[0], p_detonation[1], p_detonation[2], '烟幕', color='dimgray')


    # --- 5. Finalize and Show Plot ---
    ax.set_xlabel('X 轴 (米)')
    ax.set_ylabel('Y 轴 (米)')
    ax.set_zlabel('Z 轴 (米)')
    ax.set_title(f'问题一 (t={t_detonation:.1f}s 爆炸瞬间)', fontsize=16)
    
    # Set axis limits to show all relevant objects
    ax.set_xlim([0, 21000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([0, 2500])
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='无人机 (爆炸时刻)', markerfacecolor='r', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='导弹 (爆炸时刻)', markerfacecolor='b', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='假目标', markerfacecolor='k', markersize=10),
        mpatches.Patch(color='green', alpha=0.5, label='真目标'),
        mpatches.Patch(color='gray', alpha=0.4, label='烟幕云'),
        Line2D([0], [0], linestyle='--', color='r', label='无人机轨迹'),
        Line2D([0], [0], linestyle='--', color='b', label='导弹轨迹')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.view_init(elev=20, azim=-75)
    plt.show()