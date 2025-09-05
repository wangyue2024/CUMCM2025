# models/physics_model.py
# (中文注释版)

import numpy as np

class PhysicsModelWithCylinder:
    """
    物理模型类，集成了圆柱体目标建模和奖励塑造逻辑。
    该类负责处理所有与物理运动、空间几何和遮蔽判断相关的计算。
    """
    def __init__(self, missile_id: str, uav_id: str, config_obj: object):
        self.config = config_obj
        self.p_missile_0 = self.config.MISSILE_INITIAL_POS[missile_id]
        self.p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        
        direction_vec_m = self.config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = self.config.V_MISSILE * self.u_missile
        
        dist_to_target = np.linalg.norm(self.config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / self.config.V_MISSILE

        self.target_sample_points = self._generate_cylinder_sample_points()

    def _generate_cylinder_sample_points(self) -> np.ndarray:
        points = []
        cx, cy = self.config.CYLINDER_CENTER_XY
        radius = self.config.CYLINDER_RADIUS
        height = self.config.CYLINDER_HEIGHT
        num_edge_points = self.config.CYLINDER_SAMPLE_POINTS_EDGE
        
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

    def _get_missile_pos(self, t: float) -> np.ndarray:
        return self.p_missile_0 + self.v_vec_missile * t

    @staticmethod
    def _distance_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        if np.array_equal(seg_start, seg_end): return np.linalg.norm(point - seg_start)
        vec_seg = seg_end - seg_start
        vec_point = point - seg_start
        dot_product = np.dot(vec_point, vec_seg)
        seg_len_sq = np.dot(vec_seg, vec_seg)
        c = dot_product / seg_len_sq
        if c < 0: return np.linalg.norm(point - seg_start)
        if c > 1: return np.linalg.norm(point - seg_end)
        projection = seg_start + c * vec_seg
        return np.linalg.norm(point - projection)

    def calculate_shielding_metrics(self, uav_speed: float, uav_theta: float, launch_times: list, det_delays: list, time_step: float = None) -> tuple:
        """
        核心计算函数。
        **关键改动**: 增加可选的 time_step 参数。
        """
        # 如果未指定时间步长，则使用config中为优化设定的默认值
        sim_time_step = time_step if time_step is not None else self.config.SIM_TS_OPTIMIZATION

        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        smoke_events = []
        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            dx = v_vec_uav[0] * dt_det
            dy = v_vec_uav[1] * dt_det
            dz = -0.5 * self.config.G * dt_det**2
            p_detonation = p_launch + np.array([dx, dy, dz])
            t_detonation = t_launch + dt_det
            t_end_effective = t_detonation + self.config.T_SMOKE_EFFECTIVE
            smoke_events.append({'p_det': p_detonation, 't_det': t_detonation, 't_end': t_end_effective})

        if not smoke_events: return 0.0, float('inf'), {}

        sim_start_time = min(event['t_det'] for event in smoke_events)
        sim_end_time = min(max(event['t_end'] for event in smoke_events), self.time_to_impact)
        
        if sim_end_time <= sim_start_time: return 0.0, float('inf'), {}
        
        num_steps = int((sim_end_time - sim_start_time) / sim_time_step) + 1
        time_points = np.linspace(sim_start_time, sim_end_time, num_steps)
        shielded_time_slots = np.zeros(num_steps, dtype=bool)
        global_min_miss_distance = float('inf')

        for i, t in enumerate(time_points):
            p_missile_t = self._get_missile_pos(t)
            is_shielded_this_step = False
            
            for event in smoke_events:
                if event['t_det'] <= t < event['t_end']:
                    dt_since_det = t - event['t_det']
                    p_smoke_center_t = event['p_det'] - np.array([0, 0, self.config.V_SMOKE_SINK * dt_since_det])
                    
                    all_points_shielded = True
                    min_dist_this_event = float('inf')
                    
                    for p_target_sample in self.target_sample_points:
                        dist = self._distance_point_to_segment(p_smoke_center_t, p_missile_t, p_target_sample)
                        min_dist_this_event = min(min_dist_this_event, dist)
                        if dist > self.config.R_SMOKE:
                            all_points_shielded = False
                            break # 只要有一个点没被遮蔽，就无需再检查此烟幕的其他点
                    
                    global_min_miss_distance = min(global_min_miss_distance, min_dist_this_event)
                    
                    if all_points_shielded:
                        is_shielded_this_step = True
                        break
            
            if is_shielded_this_step:
                shielded_time_slots[i] = True

        total_shielding_time = np.sum(shielded_time_slots) * sim_time_step
        min_miss_distance = 0.0 if total_shielding_time > 0 else global_min_miss_distance
        details = {"total_shielding_time": total_shielding_time, "min_miss_distance": min_miss_distance, "smoke_events": smoke_events}
        
        return total_shielding_time, min_miss_distance, details

    def cost_function_q2(self, x: np.ndarray) -> float:
        """为问题2封装的成本函数，自动使用为优化设定的时间步长。"""
        uav_speed, uav_theta, launch_time, det_delay = x
        
        shielding_time, min_miss_distance, _ = self.calculate_shielding_metrics(
            uav_speed, uav_theta, [launch_time], [det_delay]
        )
        
        if shielding_time > 0:
            return -shielding_time
        else:
            return self.config.REWARD_SHAPING_PENALTY_FACTOR * min_miss_distance
            
    def cost_function_q3(self, x: np.ndarray) -> float:
        """为问题3封装的成本函数，处理多弹药投放策略。"""
        # 解包决策变量
        uav_speed, uav_theta = x[0], x[1]
        num_smokes = int((len(x) - 2) / 2)  # 计算烟幕弹数量
        
        # 提取投放时间和起爆延迟
        launch_times = [x[2 + i*2] for i in range(num_smokes)]
        det_delays = [x[3 + i*2] for i in range(num_smokes)]
        
        # 检查约束条件：投放时间必须递增
        for i in range(1, num_smokes):
            if launch_times[i] <= launch_times[i-1]:
                return 10.0  # 违反约束，返回惩罚值
        
        # 计算遮蔽指标
        shielding_time, min_miss_distance, _ = self.calculate_shielding_metrics(
            uav_speed, uav_theta, launch_times, det_delays
        )
        
        if shielding_time > 0:
            return -shielding_time
        else:
            return self.config.REWARD_SHAPING_PENALTY_FACTOR * min_miss_distance
            
    def calculate_shielding_metrics_detailed(self, uav_speed: float, uav_theta: float, launch_times: list, det_delays: list, time_step: float = None) -> tuple:
        """
        为问题三及以后设计的详细指标计算函数。
        除了计算总遮蔽时间，还为每个烟幕事件计算其独立的最小未命中距离。
        """
        sim_time_step = time_step if time_step is not None else self.config.SIM_TS_OPTIMIZATION

        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        smoke_events = []
        for i, (t_launch, dt_det) in enumerate(zip(launch_times, det_delays)):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            dx = v_vec_uav[0] * dt_det
            dy = v_vec_uav[1] * dt_det
            dz = -0.5 * self.config.G * dt_det**2
            p_detonation = p_launch + np.array([dx, dy, dz])
            t_detonation = t_launch + dt_det
            t_end_effective = t_detonation + self.config.T_SMOKE_EFFECTIVE
            smoke_events.append({
                'id': i,
                'p_det': p_detonation, 
                't_det': t_detonation, 
                't_end': t_end_effective,
                'min_miss_distance': float('inf'), # 为每个烟幕初始化最小未命中距离
                'contributed_to_shielding': False  # 标记是否贡献了遮蔽
            })

        if not smoke_events: 
            return 0.0, [float('inf')] * len(launch_times), {}

        sim_start_time = min(event['t_det'] for event in smoke_events)
        sim_end_time = min(max(event['t_end'] for event in smoke_events), self.time_to_impact)
      
        if sim_end_time <= sim_start_time: 
            return 0.0, [float('inf')] * len(launch_times), {}
      
        num_steps = int((sim_end_time - sim_start_time) / sim_time_step) + 1
        time_points = np.linspace(sim_start_time, sim_end_time, num_steps)
        shielded_time_slots = np.zeros(num_steps, dtype=bool)

        for i, t in enumerate(time_points):
            p_missile_t = self._get_missile_pos(t)
            
            # 计算每个烟幕的最小距离
            for event in smoke_events:
                if event['t_det'] <= t < event['t_end']:
                    dt_since_det = t - event['t_det']
                    p_smoke_center_t = event['p_det'] - np.array([0, 0, self.config.V_SMOKE_SINK * dt_since_det])
                    
                    # 计算此烟幕在此刻对所有采样点的最小距离
                    for p_target_sample in self.target_sample_points:
                        dist = self._distance_point_to_segment(p_smoke_center_t, p_missile_t, p_target_sample)
                        event['min_miss_distance'] = min(event['min_miss_distance'], dist)
            
            # 判断总体是否遮蔽
            is_shielded_this_step = True
            for p_target_sample in self.target_sample_points:
                is_sample_point_shielded = False
                contributing_smoke = None
                
                for event in smoke_events:
                    if event['t_det'] <= t < event['t_end']:
                        dt_since_det = t - event['t_det']
                        p_smoke_center_t = event['p_det'] - np.array([0, 0, self.config.V_SMOKE_SINK * dt_since_det])
                        dist = self._distance_point_to_segment(p_smoke_center_t, p_missile_t, p_target_sample)
                        if dist <= self.config.R_SMOKE:
                            is_sample_point_shielded = True
                            contributing_smoke = event
                            break
                            
                if not is_sample_point_shielded:
                    is_shielded_this_step = False
                    break
                elif contributing_smoke:
                    contributing_smoke['contributed_to_shielding'] = True
            
            if is_shielded_this_step:
                shielded_time_slots[i] = True

        total_shielding_time = np.sum(shielded_time_slots) * sim_time_step
      
        # 提取每个烟幕的最终最小未命中距离
        individual_min_miss_distances = [event['min_miss_distance'] for event in smoke_events]

        details = {
            "total_shielding_time": total_shielding_time, 
            "smoke_events": smoke_events,
            "individual_min_miss_distances": individual_min_miss_distances,
            "contributed_to_shielding": [event['contributed_to_shielding'] for event in smoke_events]
        }
      
        return total_shielding_time, individual_min_miss_distances, details

    def cost_function_q3_enhanced(self, x: np.ndarray) -> float:
        """
        为问题3设计的增强版成本函数，为每个烟幕弹提供梯度引导。
        """
        # 解包决策变量
        uav_speed, uav_theta = x[0], x[1]
        num_smokes = int((len(x) - 2) / 2)  # 计算烟幕弹数量
        
        # 提取投放时间和起爆延迟
        launch_times = [x[2 + i*2] for i in range(num_smokes)]
        det_delays = [x[3 + i*2] for i in range(num_smokes)]
        
        # 检查约束条件：投放时间必须递增
        for i in range(1, num_smokes):
            if launch_times[i] <= launch_times[i-1]:
                return 10.0  # 违反约束，返回惩罚值
        
        # 调用新的详细计算函数
        shielding_time, individual_distances, details = self.calculate_shielding_metrics_detailed(
            uav_speed, uav_theta, launch_times, det_delays
        )
        
        # 主要目标：最大化遮蔽时间
        cost = -shielding_time
        
        # 辅助目标：对每个烟幕的未命中距离进行惩罚
        # 只有当烟幕的最小距离大于烟幕半径时（即它从未成功遮蔽过任何点），才施加惩罚
        contributed = details["contributed_to_shielding"]
        for i, dist in enumerate(individual_distances):
            if not contributed[i] and dist > self.config.R_SMOKE:
                # 对未贡献遮蔽的烟幕施加距离惩罚，引导它们向有效位置移动
                cost += 0.0001 * (dist - self.config.R_SMOKE)
        
        return cost