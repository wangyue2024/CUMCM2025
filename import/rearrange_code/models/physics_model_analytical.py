# models/physics_model_analytical.py

import numpy as np
from typing import List, Tuple, Dict

class PhysicsModelAnalytical:
    """
    基于完全解析法的物理模型。
    - 使用分段求解策略处理Case 1/2切换，确保计算精度。
    - 实现“状态感知”的奖励塑造，为优化器在任何情况下提供平滑、准确的梯度。
    """
    def __init__(self, missile_id: str, config_obj: object):
        self.config = config_obj
        self.p_missile_0 = self.config.MISSILE_INITIAL_POS[missile_id]
        
        # 简化目标为圆柱中心点
        self.p_target = np.array([
            self.config.CYLINDER_CENTER_XY[0],
            self.config.CYLINDER_CENTER_XY[1],
            # self.config.CYLINDER_HEIGHT / 2.0
            0
        ])
        
        direction_vec_m = self.config.P_FALSE_TARGET - self.p_missile_0
        self.u_missile = direction_vec_m / np.linalg.norm(direction_vec_m)
        self.v_vec_missile = self.config.V_MISSILE * self.u_missile
        
        dist_to_target = np.linalg.norm(self.config.P_FALSE_TARGET - self.p_missile_0)
        self.time_to_impact = dist_to_target / self.config.V_MISSILE

    # ===================================================================
    # 核心计算模块 (私有方法)
    # ===================================================================

    def _get_all_polynomial_coefficients(self, p_det: np.ndarray, t_det: float) -> Dict[str, np.ndarray]:
        """一次性计算所有需要的多项式系数。"""
        m0, vm, p_target = self.p_missile_0, self.v_vec_missile, self.p_target
        v_sink, r_smoke = np.array([0., 0., -self.config.V_SMOKE_SINK]), self.config.R_SMOKE
        
        L0 = p_target - m0
        L1 = -vm
        S0 = p_det - v_sink * t_det - m0
        S1 = v_sink - vm

        d2, d1, d0 = np.dot(L1, L1), 2 * np.dot(L0, L1), np.dot(L0, L0)
        D_coeffs = np.array([d2, d1, d0])

        X2 = np.cross(S1, L1)
        X1 = np.cross(S0, L1) + np.cross(S1, L0)
        X0 = np.cross(S0, L0)

        q4, q3 = np.dot(X2, X2), 2 * np.dot(X2, X1)
        q2 = 2 * np.dot(X2, X0) + np.dot(X1, X1)
        q1, q0 = 2 * np.dot(X1, X0), np.dot(X0, X0)
        Q_coeffs = np.array([q4, q3, q2, q1, q0])

        R_sq = r_smoke**2
        P4_coeffs = np.polyadd(Q_coeffs, -R_sq * np.pad(D_coeffs, (2, 0), 'constant'))

        p2a_2, p2a_1 = np.dot(S1, S1), 2 * np.dot(S0, S1)
        p2a_0 = np.dot(S0, S0) - R_sq
        P2_A_coeffs = np.array([p2a_2, p2a_1, p2a_0])

        n2, n1 = np.dot(S1, L1), np.dot(S0, L1) + np.dot(S1, L0)
        n0 = np.dot(S0, L0)
        N_coeffs = np.array([n2, n1, n0])
        
        return {'P4': P4_coeffs, 'P2_A': P2_A_coeffs, 'N': N_coeffs}

    def _solve_inequality_for_intervals(self, coeffs: np.ndarray, interval: Tuple[float, float]) -> List[Tuple[float, float]]:
        """求解 P(t) <= 0，返回满足条件的区间列表。"""
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

    def _get_min_poly_val(self, coeffs: np.ndarray, interval: Tuple[float, float]) -> float:
        """计算多项式在区间内的最小值。"""
        t_start, t_end = interval
        if len(coeffs) <= 1: return np.polyval(coeffs, t_start)
        
        deriv_coeffs = np.polyder(coeffs)
        points = [t_start, t_end]
        if len(deriv_coeffs) > 0:
            roots = np.roots(deriv_coeffs)
            points.extend([r.real for r in roots if np.isreal(r) and t_start < r.real < t_end])
        
        return np.min(np.polyval(coeffs, points))

    def _calculate_metrics_for_one_smoke(self, p_det: np.ndarray, t_det: float) -> Dict:
        """为单枚烟幕弹计算遮蔽区间和代理成本。"""
        T_start = t_det
        T_end = t_det + self.config.T_SMOKE_EFFECTIVE
        
        coeffs = self._get_all_polynomial_coefficients(p_det, t_det)
        N_coeffs, P4_coeffs, P2_A_coeffs = coeffs['N'], coeffs['P4'], coeffs['P2_A']

        critical_roots = np.roots(N_coeffs)
        boundaries = sorted(list(set([T_start, T_end] + [r.real for r in critical_roots if np.isreal(r) and T_start < r.real < T_end])))

        shielding_intervals = []
        min_proxy_vals = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            t_mid = (start + end) / 2.0
            
            if np.polyval(N_coeffs, t_mid) >= 0:  # Case 1
                shielding_intervals.extend(self._solve_inequality_for_intervals(P4_coeffs, (start, end)))
                min_proxy_vals.append(self._get_min_poly_val(P4_coeffs, (start, end)))
            else:  # Case 2
                shielding_intervals.extend(self._solve_inequality_for_intervals(P2_A_coeffs, (start, end)))
                min_proxy_vals.append(self._get_min_poly_val(P2_A_coeffs, (start, end)))
        
        proxy_cost = min(min_proxy_vals) if min_proxy_vals else float('inf')
        return {"intervals": shielding_intervals, "proxy_cost": proxy_cost}

    # ===================================================================
    # 公共接口：成本函数
    # ===================================================================

    def _calculate_combined_metrics(self, uav_id: str, uav_speed: float, uav_theta: float, launch_times: list, det_delays: list) -> Dict:
        """计算多烟幕策略的总指标。"""
        p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        
        all_intervals = []
        proxy_costs = []

        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = p_uav_0 + v_vec_uav * t_launch
            p_det = p_launch + v_vec_uav * dt_det + 0.5 * np.array([0, 0, -self.config.G]) * dt_det**2
            t_det = t_launch + dt_det
            
            metrics = self._calculate_metrics_for_one_smoke(p_det, t_det)
            all_intervals.extend(metrics["intervals"])
            proxy_costs.append(metrics["proxy_cost"])

        # 合并区间
        if not all_intervals:
            merged_intervals = []
        else:
            all_intervals.sort(key=lambda x: x[0])
            merged_intervals = [all_intervals[0]]
            for start, end in all_intervals[1:]:
                last_start, last_end = merged_intervals[-1]
                if start <= last_end:
                    merged_intervals[-1] = (last_start, max(last_end, end))
                else:
                    merged_intervals.append((start, end))
        
        total_time = sum(end - start for start, end in merged_intervals)
        return {"total_time": total_time, "proxy_costs": proxy_costs}

    def cost_function_q2(self, x: np.ndarray) -> float:
        """问题2的成本函数。"""
        uav_speed, uav_theta, launch_time, det_delay = x
        metrics = self._calculate_combined_metrics('FY1', uav_speed, uav_theta, [launch_time], [det_delay])
        
        total_time = metrics["total_time"]
        if total_time > 0:
            return -total_time
        else:
            # 如果没有遮蔽，使用代理成本进行梯度引导
            proxy_cost = metrics["proxy_costs"][0]
            return self.config.REWARD_SHAPING_PENALTY_FACTOR * proxy_cost

    def cost_function_q3(self, x: np.ndarray) -> float:
        """问题3的成本函数。"""
        v_uav, theta_uav, t1, t2, t3, dt1, dt2, dt3 = x
        
        if not (t2 >= t1 + self.config.MIN_LAUNCH_INTERVAL and t3 >= t2 + self.config.MIN_LAUNCH_INTERVAL):
            penalty = 100 * (max(0, t1 + self.config.MIN_LAUNCH_INTERVAL - t2) + max(0, t2 + self.config.MIN_LAUNCH_INTERVAL - t3))
            return 100.0 + penalty # 返回一个大的惩罚值
            
        metrics = self._calculate_combined_metrics('FY1', v_uav, theta_uav, [t1, t2, t3], [dt1, dt2, dt3])
        
        total_time = metrics["total_time"]
        proxy_costs = metrics["proxy_costs"]
        
        # 对每个未产生遮蔽的烟幕弹施加惩罚
        penalty = sum(max(0, p_cost) for p_cost in proxy_costs)
        
        return -total_time + self.config.REWARD_SHAPING_PENALTY_FACTOR * penalty

    def get_final_details(self, uav_id: str, x: np.ndarray) -> Dict:
        """根据最优解向量，计算用于报告和可视化的详细信息。"""
        num_smokes = (len(x) - 2) // 2
        uav_speed, uav_theta = x[0], x[1]
        launch_times = x[2 : 2 + num_smokes]
        det_delays = x[2 + num_smokes :]

        p_uav_0 = self.config.UAV_INITIAL_POS[uav_id]
        v_vec_uav = np.array([uav_speed * np.cos(uav_theta), uav_speed * np.sin(uav_theta), 0])
        
        smoke_events = []
        for t_launch, dt_det in zip(launch_times, det_delays):
            p_launch = p_uav_0 + v_vec_uav * t_launch
            p_det = p_launch + v_vec_uav * dt_det + 0.5 * np.array([0, 0, -self.config.G]) * dt_det**2
            t_det = t_launch + dt_det
            smoke_events.append({'p_det': p_det, 't_det': t_det, 't_end': t_det + self.config.T_SMOKE_EFFECTIVE})
        
        metrics = self._calculate_combined_metrics(uav_id, uav_speed, uav_theta, launch_times, det_delays)

        return {
            "total_shielding_time": metrics["total_time"],
            "smoke_events": smoke_events,
            "uav_params": {"speed": uav_speed, "theta_rad": uav_theta}
        }