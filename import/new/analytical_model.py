# analytical_model.py

import numpy as np
from typing import List, Tuple, Dict
from config import Config

class PhysicsModelAnalytical:
    """
    完全基于解析法构建的物理模型。
    该模型负责计算精确的遮蔽时间，并为优化器提供高性能的代价函数。
    """
    def __init__(self, missile_id: str, uav_id: str):
        self.config = Config()
        self.p_missile_0 = self.config.MISSILE_INITIAL_POS[missile_id]
        self.p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        self.p_target = self.config.P_TRUE_TARGET
        
        direction_vec_m = self.config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = self.config.V_MISSILE * self.u_missile
        
        dist_to_target = np.linalg.norm(self.config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / self.config.V_MISSILE

    # ===================================================================
    # 核心解析计算模块 (私有方法)
    # ===================================================================
    
    def _get_all_polynomial_coefficients(self, p_det: np.ndarray, t_det: float) -> Dict[str, np.ndarray]:
        """一次性计算单个烟幕所需的所有多项式系数。"""
        m0, vm, p_target = self.p_missile_0, self.v_vec_missile, self.p_target
        v_sink, r_smoke = self.config.V_SMOKE_SINK, self.config.R_SMOKE
        
        L0 = p_target - m0
        L1 = -vm
        S0 = p_det - v_sink * t_det - m0
        S1 = v_sink - vm

        d2 = np.dot(L1, L1); d1 = 2 * np.dot(L0, L1); d0 = np.dot(L0, L0)
        D_coeffs = np.array([d2, d1, d0])

        X2 = np.cross(S1, L1); X1 = np.cross(S0, L1) + np.cross(S1, L0); X0 = np.cross(S0, L0)
        
        q4 = np.dot(X2, X2); q3 = 2 * np.dot(X2, X1)
        q2 = 2 * np.dot(X2, X0) + np.dot(X1, X1); q1 = 2 * np.dot(X1, X0); q0 = np.dot(X0, X0)
        Q_coeffs = np.array([q4, q3, q2, q1, q0])

        R_sq = r_smoke**2
        P4_coeffs = np.polyadd(Q_coeffs, -R_sq * np.pad(D_coeffs, (2, 0), 'constant'))

        p2a_2 = np.dot(S1, S1); p2a_1 = 2 * np.dot(S0, S1); p2a_0 = np.dot(S0, S0) - R_sq
        P2_A_coeffs = np.array([p2a_2, p2a_1, p2a_0])

        n2 = np.dot(S1, L1); n1 = np.dot(S0, L1) + np.dot(S1, L0); n0 = np.dot(S0, L0)
        N_coeffs = np.array([n2, n1, n0])
        
        return {'P4': P4_coeffs, 'P2_A': P2_A_coeffs, 'N': N_coeffs}

    def _solve_inequality_for_intervals(self, coeffs: np.ndarray, interval: Tuple[float, float]) -> List[Tuple[float, float]]:
        """求解多项式 P(t) <= 0，返回满足条件的区间列表。"""
        t_start, t_end = interval
        if t_start >= t_end: return []
        if len(coeffs) == 1: return [(t_start, t_end)] if coeffs[0] <= 0 else []
        
        roots = np.roots(coeffs)
        points = sorted(list(set([t_start, t_end] + [r.real for r in roots if np.isreal(r) and t_start < r.real < t_end])))
        
        intervals = []
        for i in range(len(points) - 1):
            p_start, p_end = points[i], points[i+1]
            if np.polyval(coeffs, (p_start + p_end) / 2.0) <= 0:
                intervals.append((p_start, p_end))
        return intervals

    def _calculate_single_smoke_intervals_robust(self, p_det: np.ndarray, t_det: float) -> List[Tuple[float, float]]:
        """鲁棒地计算单个烟幕的遮蔽区间列表（处理Case1/2切换）。"""
        T_start = t_det
        T_end = t_det + self.config.T_SMOKE_EFFECTIVE
        
        coeffs = self._get_all_polynomial_coefficients(p_det, t_det)
        N_coeffs = coeffs['N']
        
        critical_roots = np.roots(N_coeffs)
        boundaries = sorted(list(set([T_start, T_end] + [r.real for r in critical_roots if np.isreal(r) and T_start < r.real < T_end])))
        
        shielded_intervals = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            t_mid = (start + end) / 2.0
            
            case_coeffs = coeffs['P4'] if np.polyval(N_coeffs, t_mid) >= 0 else coeffs['P2_A']
            shielded_intervals.extend(self._solve_inequality_for_intervals(case_coeffs, (start, end)))
            
        return shielded_intervals

    def _get_min_poly_value(self, coeffs: np.ndarray, interval: Tuple[float, float]) -> float:
        """计算多项式在区间内的最小值，用于奖励塑造。"""
        t_start, t_end = interval
        if len(coeffs) <= 1: return np.polyval(coeffs, t_start)
        
        deriv_coeffs = np.polyder(coeffs)
        points = [t_start, t_end] + [p.real for p in np.roots(deriv_coeffs) if np.isreal(p) and t_start < p.real < t_end]
        return np.min(np.polyval(coeffs, points))

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并重叠区间。"""
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    # ===================================================================
    # 面向求解器的代价函数接口
    # ===================================================================

    def cost_function_q2(self, x: np.ndarray) -> float:
        """问题二的代价函数：单无人机，单弹药。"""
        uav_speed, uav_theta, t_launch, dt_det = x
        
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        p_launch = self.p_uav_0 + v_vec_uav * t_launch
        p_det = p_launch + v_vec_uav * dt_det + 0.5 * np.array([0, 0, -self.config.G]) * dt_det**2
        t_det = t_launch + dt_det

        intervals = self._calculate_single_smoke_intervals_robust(p_det, t_det)
        total_time = sum(end - start for start, end in intervals)

        if total_time > 0:
            return -total_time
        else:
            # 奖励塑造：如果没有遮蔽，则惩罚“离成功有多远”
            coeffs = self._get_all_polynomial_coefficients(p_det, t_det)['P4']
            min_p_val = self._get_min_poly_value(coeffs, (t_det, t_det + self.config.T_SMOKE_EFFECTIVE))
            return self.config.REWARD_SHAPING_PENALTY_FACTOR * max(0, min_p_val)

    def cost_function_q3(self, x: np.ndarray) -> float:
        """问题三的代价函数：单无人机，三弹药，带约束和奖励塑造。"""
        v_uav, theta_uav, t1, t2, t3, dt1, dt2, dt3 = x
        
        # 1. 检查投放间隔约束
        if not (t2 >= t1 + self.config.MIN_LAUNCH_INTERVAL and t3 >= t2 + self.config.MIN_LAUNCH_INTERVAL):
            penalty = 100 * (max(0, t1 + self.config.MIN_LAUNCH_INTERVAL - t2) + max(0, t2 + self.config.MIN_LAUNCH_INTERVAL - t3))
            return 100 + penalty # 返回一个大的惩罚值

        # 2. 计算每个烟幕的遮蔽区间和代理成本
        v_vec_uav = np.array([v_uav * np.cos(theta_uav), v_uav * np.sin(theta_uav), 0])
        launch_times = [t1, t2, t3]
        det_delays = [dt1, dt2, dt3]
        
        all_intervals = []
        total_proxy_penalty = 0.0

        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = self.p_uav_0 + v_vec_uav * t_launch
            p_det = p_launch + v_vec_uav * dt_det + 0.5 * np.array([0, 0, -self.config.G]) * dt_det**2
            t_det = t_launch + dt_det
            
            intervals = self._calculate_single_smoke_intervals_robust(p_det, t_det)
            all_intervals.extend(intervals)
            
            # 奖励塑造：为每个未产生遮蔽的烟幕计算惩罚
            if not intervals:
                coeffs = self._get_all_polynomial_coefficients(p_det, t_det)['P4']
                min_p_val = self._get_min_poly_value(coeffs, (t_det, t_det + self.config.T_SMOKE_EFFECTIVE))
                total_proxy_penalty += max(0, min_p_val)

        # 3. 合并区间并计算总成本
        merged = self._merge_intervals(all_intervals)
        total_time = sum(end - start for start, end in merged)
        
        final_cost = -total_time + self.config.REWARD_SHAPING_PENALTY_FACTOR * total_proxy_penalty
        return final_cost