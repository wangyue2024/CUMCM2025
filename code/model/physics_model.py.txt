# models/physics_model.py
# (中文注释版)

import numpy as np
# 假设config.py在同一个目录下或已安装为包
import config

class PhysicsModelWithCylinder:
    """
    物理模型类，集成了圆柱体目标建模和奖励塑造逻辑。
    
    该类负责处理所有与物理运动、空间几何和遮蔽判断相关的计算。
    它被设计为可重用的模块，为上层的优化器提供清晰的接口。
    """
    def __init__(self, missile_id: str, uav_id: str, config_obj: object):
        """
        初始化物理模型实例。
        
        Args:
            missile_id (str): 要模拟的导弹编号 (例如, 'M1').
            uav_id (str): 要模拟的无人机编号 (例如, 'FY1').
            config_obj (object): 包含所有配置参数的配置对象.
        """
        self.config = config_obj
        self.p_missile_0 = self.config.MISSILE_INITIAL_POS[missile_id]
        self.p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        
        # 预计算导弹轨迹的常数，避免重复计算
        direction_vec_m = self.config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = self.config.V_MISSILE * self.u_missile
        
        # 预计算导弹击中假目标所需的时间，作为模拟的上限
        dist_to_target = np.linalg.norm(self.config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / self.config.V_MISSILE

        # 预计算真目标（圆柱体）的表面采样点
        self.target_sample_points = self._generate_cylinder_sample_points()

    def _generate_cylinder_sample_points(self) -> np.ndarray:
        """
        生成圆柱体真目标表面的代表性采样点。
        
        Returns:
            np.ndarray: 一个 (N, 3) 的数组，每行是一个采样点的 [x, y, z] 坐标。
        """
        points = []
        cx, cy = self.config.CYLINDER_CENTER_XY
        radius = self.config.CYLINDER_RADIUS
        height = self.config.CYLINDER_HEIGHT
        num_edge_points = self.config.CYLINDER_SAMPLE_POINTS_EDGE
        
        # 添加上、下底面的中心点
        points.append(np.array([cx, cy, 0]))
        points.append(np.array([cx, cy, height]))
        
        # 沿圆周生成边缘点
        for i in range(num_edge_points):
            angle = 2 * np.pi * i / num_edge_points
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            points.append(np.array([x, y, 0]))          # 下底面边缘
            points.append(np.array([x, y, height]))       # 上底面边缘
            points.append(np.array([x, y, height / 2.0])) # 侧面中部
            
        return np.array(points)

    def _get_missile_pos(self, t: float) -> np.ndarray:
        """计算导弹在时刻 t 的位置。"""
        return self.p_missile_0 + self.v_vec_missile * t

    @staticmethod
    def _distance_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """计算空间中一个点到一条线段的最短距离。"""
        if np.array_equal(seg_start, seg_end):
            return np.linalg.norm(point - seg_start)
        
        vec_seg = seg_end - seg_start
        vec_point = point - seg_start
        
        dot_product = np.dot(vec_point, vec_seg)
        seg_len_sq = np.dot(vec_seg, vec_seg)
        
        # 找到点在线段方向上的投影比例
        c = dot_product / seg_len_sq
        
        # 根据投影位置判断最近点是端点还是投影点
        if c < 0: return np.linalg.norm(point - seg_start)
        if c > 1: return np.linalg.norm(point - seg_end)
        
        projection = seg_start + c * vec_seg
        return np.linalg.norm(point - projection)

    def calculate_shielding_metrics(self, uav_speed: float, uav_theta: float, launch_times: list, det_delays: list) -> tuple:
        """
        核心计算函数。根据给定的策略，模拟整个过程并返回关键性能指标。
        
        Args:
            uav_speed (float): 无人机飞行速度 (m/s).
            uav_theta (float): 无人机飞行方向角 (弧度).
            launch_times (list): 干扰弹的投放时间列表 [s].
            det_delays (list): 干扰弹的起爆延迟列表 [s].
            
        Returns:
            tuple: (total_shielding_time, min_miss_distance, details_dict)
                   - total_shielding_time (float): 总有效遮蔽时长 (s).
                   - min_miss_distance (float): 全局最小未命中距离 (m). 如果有遮蔽则为0.
                   - details_dict (dict): 包含模拟详细信息的字典，用于分析和可视化.
        """
        # 1. 计算无人机速度向量和所有烟幕事件的时空信息
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        smoke_events = []
        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            # 平抛运动位移
            dx = v_vec_uav[0] * dt_det
            dy = v_vec_uav[1] * dt_det
            dz = -0.5 * self.config.G * dt_det**2
            p_detonation = p_launch + np.array([dx, dy, dz])
            
            t_detonation = t_launch + dt_det
            t_end_effective = t_detonation + self.config.T_SMOKE_EFFECTIVE
            smoke_events.append({
                'p_det': p_detonation, 
                't_det': t_detonation, 
                't_end': t_end_effective
            })

        if not smoke_events:
            return 0.0, float('inf'), {}

        # 2. 设置模拟的时间范围和步长
        sim_start_time = min(event['t_det'] for event in smoke_events)
        sim_end_time = min(max(event['t_end'] for event in smoke_events), self.time_to_impact)
        
        num_steps = int((sim_end_time - sim_start_time) / self.config.SIMULATION_TIME_STEP) + 1
        if num_steps <= 0: return 0.0, float('inf'), {}

        time_points = np.linspace(sim_start_time, sim_end_time, num_steps)
        shielded_time_slots = np.zeros(num_steps, dtype=bool)
        global_min_miss_distance = float('inf')

        # 3. 遍历时间步，进行模拟和判断
        for i, t in enumerate(time_points):
            p_missile_t = self._get_missile_pos(t)
            is_shielded_this_step = False
            
            for event in smoke_events:
                if event['t_det'] <= t < event['t_end']:
                    dt_since_det = t - event['t_det']
                    p_smoke_center_t = event['p_det'] - np.array([0, 0, self.config.V_SMOKE_SINK * dt_since_det])
                    
                    all_points_shielded = True
                    min_dist_this_event = float('inf')
                    
                    # 检查所有目标采样点
                    for p_target_sample in self.target_sample_points:
                        dist = self._distance_point_to_segment(p_smoke_center_t, p_missile_t, p_target_sample)
                        min_dist_this_event = min(min_dist_this_event, dist)
                        if dist > self.config.R_SMOKE:
                            all_points_shielded = False
                    
                    global_min_miss_distance = min(global_min_miss_distance, min_dist_this_event)
                    
                    if all_points_shielded:
                        is_shielded_this_step = True
                        break  # 已被此烟幕遮蔽，无需检查本时间步的其他烟幕
            
            if is_shielded_this_step:
                shielded_time_slots[i] = True

        # 4. 计算最终结果
        total_shielding_time = np.sum(shielded_time_slots) * self.config.SIMULATION_TIME_STEP
        
        min_miss_distance = 0.0 if total_shielding_time > 0 else global_min_miss_distance

        details = {
            "total_shielding_time": total_shielding_time,
            "min_miss_distance": min_miss_distance,
            "smoke_events": smoke_events,
            # 可以按需添加更多细节用于可视化
        }
        
        return total_shielding_time, min_miss_distance, details

    def cost_function_q2(self, x: np.ndarray) -> float:
        """
        为问题2封装的成本函数，暴露给求解器。
        该函数实现了奖励塑造逻辑。
        
        Args:
            x (np.ndarray): 决策变量向量 [uav_speed, uav_theta, launch_time, det_delay].
            
        Returns:
            float: 成本值。值越小，代表策略越优。
        """
        uav_speed, uav_theta, launch_time, det_delay = x
        
        shielding_time, min_miss_distance, _ = self.calculate_shielding_metrics(
            uav_speed, uav_theta, [launch_time], [det_delay]
        )
        
        # 奖励塑造（Reward Shaping）逻辑
        if shielding_time > 0:
            # 主要目标：最大化遮蔽时间，即最小化其负值
            return -shielding_time
        else:
            # 辅助目标：当没有遮蔽时，最小化未命中距离
            # 返回一个小的正惩罚值，引导搜索方向
            return self.config.REWARD_SHAPING_PENALTY_FACTOR * min_miss_distance

# ==============================================================================
# Part 3: 示例调用 (用于单元测试)
# ==============================================================================
if __name__ == '__main__':
    print("--- [单元测试] 测试 models/physics_model.py ---")
    
    # 创建一个临时的Config类实例用于测试
    cfg = config.Config()
    
    # 初始化模型
    model = PhysicsModelWithCylinder(missile_id='M1', uav_id='FY1', config_obj=cfg)
    print(f"模型初始化成功，目标采样点数: {len(model.target_sample_points)}")
    print(f"导弹预计撞击时间: {model.time_to_impact:.2f} s")

    # 测试场景1: 一个随机策略
    strategy1 = [100, np.pi, 5.0, 4.0] 
    cost1 = model.cost_function_q2(strategy1)
    print(f"\n测试策略1 {strategy1} 的成本值: {cost1:.6f}")

    # 测试场景2: 另一个随机策略
    strategy2 = [70, 0, 20.0, 10.0]
    cost2 = model.cost_function_q2(strategy2)
    print(f"测试策略2 {strategy2} 的成本值: {cost2:.6f}")

    # 比较成本值，cost1和cost2应该不同，且能反映出哪个策略“更有希望”
    if cost1 != cost2:
        print("\n测试通过：成本函数能够为不同策略生成不同的成本值。")
    else:
        print("\n测试警告：不同策略产生了相同的成本值，请检查逻辑。")