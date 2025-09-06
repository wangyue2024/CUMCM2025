# solvers_p5/optimization_wrapper_p5.py

import numpy as np
from typing import List, Tuple
from models.physics_model_analytical import PhysicsModelAnalytical

class P5OptimizationWrapper:
    """
    为问题五的下层战术求解器封装优化问题。
    处理8维固定决策向量的归一化、解码和约束。
    【修正版：完全移除硬约束，依赖平滑方法】
    """
    def __init__(self, model: PhysicsModelAnalytical, uav_id: str, task_list: List[Tuple[int, str]]):
        self.model = model
        self.cfg = model.config
        self.uav_id = uav_id
        self.task_list = task_list
        
        self.dim = 8
        self.bounds_norm = [[0.0] * self.dim, [1.0] * self.dim]

    def decode(self, x_norm: np.ndarray) -> np.ndarray:
        """将归一化的8维向量解码为真实的物理参数向量。"""
        # 1. 解码飞行参数 (v, theta)
        v = self.cfg.V_UAV_MIN + x_norm[0] * (self.cfg.V_UAV_MAX - self.cfg.V_UAV_MIN)
        theta = x_norm[1] * 2 * np.pi

        # 2. 解码投放时间 (t_launch_0, t_launch_1, t_launch_2)
        # 通过变量代换和范围映射，确保时间总在合理范围内
        # 确保最晚的投弹+引爆时间也不会超过导弹撞击时间
        # 假设最长引爆延迟为20s，最小间隔为1s
        max_total_duration = 20.0 * 3 + self.cfg.MIN_LAUNCH_INTERVAL * 2
        # 规划时间窗口，留出足够的执行余量
        planning_window_end = self.model.time_to_impact - 22 
        
        t_start = x_norm[2] * planning_window_end if planning_window_end > 0 else 0.1
        
        # 允许的额外间隔范围，例如0-5秒
        delta_t1 = x_norm[3] * 5.0
        delta_t2 = x_norm[4] * 5.0
        
        t0 = t_start
        t1 = t0 + self.cfg.MIN_LAUNCH_INTERVAL + delta_t1
        t2 = t1 + self.cfg.MIN_LAUNCH_INTERVAL + delta_t2
        
        # 3. 解码引爆延迟 (dt_det_0, dt_det_1, dt_det_2)
        dt_det0 = 0.1 + x_norm[5] * (20.0 - 0.1)
        dt_det1 = 0.1 + x_norm[6] * (20.0 - 0.1)
        dt_det2 = 0.1 + x_norm[7] * (20.0 - 0.1)
        
        # 4. 组装成最终的物理向量
        final_phys_vec = np.array([
            v, theta,
            t0, dt_det0,
            t1, dt_det1,
            t2, dt_det2
        ])
        
        return final_phys_vec

    def cost_function(self, x_norm: np.ndarray) -> float:
        """
        优化器调用的成本函数接口。
        【修正版：移除所有硬约束检查，直接调用模型】
        """
        x_phys = self.decode(x_norm)
        return self.model.cost_function_p5_scalar(x_phys, self.uav_id, self.task_list)