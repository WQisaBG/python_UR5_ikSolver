import math
import numpy as np
from typing import List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter


def invTransform(Transform: np.ndarray) -> np.ndarray:
    T = np.matrix(Transform)
    R_mat = T[0:3, 0:3]
    t = T[0:3, 3]

    inverseT = np.hstack((R_mat.transpose(), -R_mat.transpose().dot(t)))
    inverseT = np.vstack((inverseT, [0, 0, 0, 1]))
    return np.asarray(inverseT)


# D-H变换矩阵
def transformDHParameter(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    T = np.array([
        [math.cos(theta), -math.sin(theta) * math.cos(alpha), math.sin(theta) * math.sin(alpha), a * math.cos(theta)],
        [math.sin(theta), math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha), a * math.sin(theta)],
        [0, math.sin(alpha), math.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T


# 前向运动学函数
def transformRobotParameter(theta: List[float]) -> np.ndarray:
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    a = [0, -0.425, -0.39225, 0, 0, 0]
    alpha = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]
    T = np.eye(4)
    for i in range(6):
        T = T.dot(transformDHParameter(a[i], d[i], alpha[i], theta[i]))
    return T


class InverseKinematicsUR5:
    def __init__(self):
        # Debug mode
        self.debug: bool = False

        self.perturb_if_singular: bool = True
        self.perturb_amount: float = 0.05
        self.previous_theta: Optional[np.ndarray] = None  # 上一次计算得到的theta

        # DH parameters
        self.d: List[float] = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.a: List[float] = [0, -0.425, -0.39225, 0, 0, 0]
        self.alpha: List[float] = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]

        # End-effector orientation offset
        self.ee_offset: np.ndarray = np.eye(4)

        # Joint limits
        self.limit_max: float = 2 * math.pi
        self.limit_min: float = -2 * math.pi

        # Joint weights
        self.joint_weights: np.ndarray = np.array([1, 1, 1, 1, 1, 1])

        # Target transformation
        self.gd: np.ndarray = np.identity(4)

        # Flag to stop IK calculation
        self.stop_flag: bool = False

        # 解析解相关变量
        self.theta1: np.ndarray = np.zeros(2)
        self.flags1: Optional[np.ndarray] = None

        self.theta5: np.ndarray = np.zeros((2, 2))
        self.flags5: Optional[np.ndarray] = None

        self.theta6: np.ndarray = np.zeros((2, 2))

        self.theta2: np.ndarray = np.zeros((2, 2, 2))
        self.theta3: np.ndarray = np.zeros((2, 2, 2))
        self.flags3: Optional[np.ndarray] = None

        self.theta4: np.ndarray = np.zeros((2, 2, 2))

        # 缓存
        self.pose_cache = {}
        self._connectivity_cache = {}

    def enableDebugMode(self, debug: bool = True) -> None:
        self.debug = debug

    def setJointLimits(self, limit_min: float, limit_max: float) -> None:
        self.limit_max = limit_max
        self.limit_min = limit_min

    def setJointWeights(self, weights: List[float]) -> None:
        self.joint_weights = np.array(weights)

    def setEERotationOffset(self, r_offset_3x3: np.ndarray) -> None:
        self.ee_offset[0:3, 0:3] = r_offset_3x3

    def setEERotationOffsetROS(self) -> None:
        r_offset_3x3 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        self.setEERotationOffset(r_offset_3x3)

    def normalize(self, value: float) -> float:
        normalized = value
        while normalized > self.limit_max:
            normalized -= 2 * math.pi
        while normalized < self.limit_min:
            normalized += 2 * math.pi
        return normalized

    def getFlags(self, nominator: float, denominator: float) -> bool:
        if denominator == 0:
            return False
        return abs(nominator / denominator) < 1.01

    def getTheta1(self) -> None:
        self.flags1 = np.ones(2)
        p05 = self.gd @ np.array([0, 0, -self.d[5], 1]) - np.array([0, 0, 0, 1])
        psi = math.atan2(p05[1], p05[0])
        L = math.sqrt(p05[0] ** 2 + p05[1] ** 2)

        if abs(self.d[3]) > L:
            if self.debug:
                print(f'L1 = {L}, denominator = {self.d[3]}')
            self.flags1[:] = self.getFlags(self.d[3], L)
            L = abs(self.d[3])
        phi = math.acos(self.d[3] / L)

        self.theta1[0] = self.normalize(psi + phi + math.pi / 2)
        self.theta1[1] = self.normalize(psi - phi + math.pi / 2)

        self.stop_flag = not np.any(self.flags1)
        if self.debug:
            print(f't1: {self.theta1}')
            print(f'flags1: {self.flags1}')

    def getTheta5(self) -> None:
        self.flags5 = np.ones((2, 2))
        p06 = self.gd[0:3, 3]

        for i in range(2):
            p16z = p06[0] * math.sin(self.theta1[i]) - p06[1] * math.cos(self.theta1[i])
            L = self.d[5]

            if abs(p16z - self.d[3]) > L:
                if self.debug:
                    print(f'L5 = {L}, denominator = {abs(p16z - self.d[3])}')
                self.flags5[i, :] = self.getFlags(p16z - self.d[3], self.d[5])
                L = abs(p16z - self.d[3])
            theta5i = math.acos((p16z - self.d[3]) / L)
            self.theta5[i, 0] = theta5i
            self.theta5[i, 1] = -theta5i

        self.stop_flag = not np.any(self.flags5)
        if self.debug:
            print(f't5: {self.theta5}')
            print(f'flags5: {self.flags5}')

    def getTheta6(self) -> None:
        for i in range(2):
            T1 = transformDHParameter(self.a[0], self.d[0], self.alpha[0], self.theta1[i])
            T61 = invTransform(invTransform(T1) @ self.gd)
            for j in range(2):
                if math.sin(self.theta5[i, j]) == 0:
                    if self.debug:
                        print("Singular case. selected theta 6 = 0")
                    self.theta6[i, j] = 0
                else:
                    self.theta6[i, j] = math.atan2(
                        -T61[1, 2] / math.sin(self.theta5[i, j]),
                         T61[0, 2] / math.sin(self.theta5[i, j]),
                    )

    def getTheta23(self) -> None:
        self.flags3 = np.ones((2, 2, 2))
        for i in range(2):
            T1 = transformDHParameter(self.a[0], self.d[0], self.alpha[0], self.theta1[i])
            T16 = invTransform(T1) @ self.gd

            for j in range(2):
                T45 = transformDHParameter(self.a[4], self.d[4], self.alpha[4], self.theta5[i, j])
                T56 = transformDHParameter(self.a[5], self.d[5], self.alpha[5], self.theta6[i, j])
                T14 = T16 @ invTransform(T45 @ T56)

                P13 = T14 @ np.array([0, -self.d[3], 0, 1]) - np.array([0, 0, 0, 1])
                L = (P13 @ P13.T) - self.a[1] ** 2 - self.a[2] ** 2

                if abs(L / (2 * self.a[1] * self.a[2])) > 1:
                    if self.debug:
                        print(f'L3 = {L}, denominator = {2 * self.a[1] * self.a[2]}')
                    self.flags3[i, j, :] = self.getFlags(L, 2 * self.a[1] * self.a[2])
                    L = math.copysign(2 * self.a[1] * self.a[2], L)

                try:
                    val = min(max(L / (2 * self.a[1] * self.a[2]), -1.0), 1.0)
                    theta3_pos = math.acos(val)
                    theta3_neg = -theta3_pos
                except ValueError:
                    theta3_pos = math.nan
                    theta3_neg = math.nan

                self.theta3[i, j, 0] = theta3_pos
                self.theta2[i, j, 0] = (
                    -math.atan2(P13[1], -P13[0])
                    + math.asin(self.a[2] * math.sin(theta3_pos) / np.linalg.norm(P13))
                )
                self.theta3[i, j, 1] = theta3_neg
                self.theta2[i, j, 1] = (
                    -math.atan2(P13[1], -P13[0])
                    + math.asin(self.a[2] * math.sin(theta3_neg) / np.linalg.norm(P13))
                )

        self.stop_flag = not np.any(self.flags3)
        if self.debug:
            print(f't2: {self.theta2}')
            print(f't3: {self.theta3}')
            print(f'flags3: {self.flags3}')

    def getTheta4(self) -> None:
        for i in range(2):
            T1 = transformDHParameter(self.a[0], self.d[0], self.alpha[0], self.theta1[i])
            T16 = invTransform(T1) @ self.gd

            for j in range(2):
                T45 = transformDHParameter(self.a[4], self.d[4], self.alpha[4], self.theta5[i, j])
                T56 = transformDHParameter(self.a[5], self.d[5], self.alpha[5], self.theta6[i, j])
                T14 = T16 @ invTransform(T45 @ T56)

                for k in range(2):
                    T13 = transformDHParameter(self.a[1], self.d[1], self.alpha[1], self.theta2[i, j, k]) @ \
                           transformDHParameter(self.a[2], self.d[2], self.alpha[2], self.theta3[i, j, k])
                    T34 = invTransform(T13) @ T14
                    self.theta4[i, j, k] = math.atan2(T34[1, 0], T34[0, 0])

        if self.debug:
            print(f't4: {self.theta4}')

    def countValidSolution(self) -> int:
        number_of_solution = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if self.flags1[i] and self.flags3[i, j, k] and self.flags5[i, j]:
                        number_of_solution += 1
        return number_of_solution

    def solveIK(self, forward_kinematics: np.ndarray) -> Optional[np.ndarray]:
        self.gd = forward_kinematics @ self.ee_offset
        self.getTheta1()
        self.getTheta5()
        self.getTheta6()
        self.getTheta23()
        self.getTheta4()
        number_of_solution = self.countValidSolution()

        if self.stop_flag or number_of_solution < 1:
            if self.debug:
                print('No solution')
            return None

        Q = np.zeros((number_of_solution, 6))
        index = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if not (self.flags1[i] and self.flags3[i, j, k] and self.flags5[i, j]):
                        continue
                    Q[index, 0] = self.normalize(self.theta1[i])
                    Q[index, 1] = self.normalize(self.theta2[i, j, k])
                    Q[index, 2] = self.normalize(self.theta3[i, j, k])
                    Q[index, 3] = self.normalize(self.theta4[i, j, k])
                    Q[index, 4] = self.normalize(self.theta5[i, j])
                    Q[index, 5] = self.normalize(self.theta6[i, j])
                    index += 1

        if self.debug:
            print(f'Number of solutions: {number_of_solution}')
            print(Q)

        return Q

    # 正向运动学
    def fk(self, theta: List[float]) -> np.ndarray:
        return transformRobotParameter(theta)

    # 优化目标函数
    def objective_func(self, theta: np.ndarray, target_pose: np.ndarray, previous_theta: Optional[np.ndarray] = None) -> float:
        T = self.fk(theta)
        pos_error = T[:3, 3] - target_pose[:3, 3]
        orient_error = R.from_matrix(T[:3, :3]).as_rotvec() - R.from_matrix(target_pose[:3, :3]).as_rotvec()

        weight_pos = 1.0
        weight_orient = 0.5
        weight_smooth = 0.1

        error = (
            weight_pos * np.linalg.norm(pos_error) +
            weight_orient * np.linalg.norm(orient_error)
        )

        if previous_theta is not None:
            error += weight_smooth * np.linalg.norm(theta - previous_theta)

        return error

    def computeJacobian(self, theta: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        J = np.zeros((6, 6))
        for i in range(6):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += delta
            theta_minus[i] -= delta

            T_plus = self.fk(theta_plus)
            T_minus = self.fk(theta_minus)

            dT = T_plus @ invTransform(T_minus)
            J[:, i] = [
                dT[0, 3], dT[1, 3], dT[2, 3],
                math.atan2(dT[2, 1], dT[2, 2]),  # x-axis rotation
                math.atan2(-dT[2, 0], math.hypot(dT[2, 1], dT[2, 2])),  # y-axis
                math.atan2(dT[1, 0], dT[0, 0])   # z-axis
            ]
        return J / (2 * delta)

    def isSingular(self, theta: np.ndarray, threshold: float = 1e-3) -> bool:
        J = self.computeJacobian(theta)
        _, S, _ = np.linalg.svd(J)
        if self.debug:
            print(f"最小奇异值: {np.min(S)}")
        return np.min(S) < threshold

    def isPoseReachable(self, target_pose: np.ndarray, num_samples: int = 50, threshold: float = 1e-3) -> bool:
        cache_key = tuple(target_pose.flatten().round(4))
        if hasattr(self, 'pose_cache') and cache_key in self.pose_cache:
            return self.pose_cache[cache_key]

        for _ in range(num_samples):
            x0 = np.random.uniform(self.limit_min, self.limit_max, size=6)
            result = minimize(
                fun=lambda x: self.objective_func(x, target_pose),
                x0=x0,
                method='L-BFGS-B',
                bounds=[(self.limit_min, self.limit_max)] * 6,
                tol=1e-5,
                options={'maxiter': 100, 'disp': False}
            )
            if result.success and np.linalg.norm(result.fun) < 0.01:
                if not hasattr(self, 'pose_cache'):
                    self.pose_cache = {}
                self.pose_cache[cache_key] = True
                return True

        if not hasattr(self, 'pose_cache'):
            self.pose_cache = {}
        self.pose_cache[cache_key] = False
        return False

    def solveIKNumerical_Scipy(
        self,
        forward_kinematics: np.ndarray,
        current_joint_angles: List[float],
        max_iter: int = 1000,
        tol: float = 1e-6,
        singular_threshold: float = 1e-6
    ) -> Optional[np.ndarray]:
        x0 = np.array(current_joint_angles).copy()
        bounds = [(self.limit_min, self.limit_max)] * 6
        self.previous_theta = x0

        if self.isSingular(x0, threshold=singular_threshold):
            print("❌ 初始构型处于奇异状态，可能导致不可达")
            return None

        result = minimize(
            fun=lambda x: self.objective_func(x, forward_kinematics),
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            tol=tol,
            options={'maxiter': max_iter, 'disp': False}
        )

        final_theta = result.x
        if self.isSingular(final_theta, threshold=singular_threshold):
            print("❌ 最终构型处于奇异状态，结果可能不稳定")
            return None

        if result.success:
            print(f"✅ 成功收敛于 {result.nit} 次迭代")
            self.previous_theta = final_theta.copy()
            return final_theta
        else:
            print("⚠️ 未成功收敛")
            print("最终误差:", result.fun)
            return None

    def findClosestIK(self, forward_kinematics: np.ndarray, current_joint_configuration: List[float], use_numerical: bool = False) -> Optional[np.ndarray]:
        if not self.isPoseReachable(forward_kinematics):
            print("❌ 目标位姿不可达")
            return None

        if use_numerical:
            Q_analytical = self.solveIK(forward_kinematics)
            if Q_analytical is not None:
                current_joint = np.array(current_joint_configuration)
                delta_Q = np.absolute(Q_analytical - current_joint) * self.joint_weights
                closest_idx = np.argmin(np.sum(delta_Q, axis=1))
                analytical_theta = Q_analytical[closest_idx]
                numerical_result = self.solveIKNumerical_Scipy(forward_kinematics, analytical_theta.tolist())
                if numerical_result is None and self.perturb_if_singular:
                    print("🔄 尝试加入扰动重新求解")
                    perturbed = [x + np.random.uniform(-self.perturb_amount, self.perturb_amount) for x in analytical_theta.tolist()]
                    numerical_result = self.solveIKNumerical_Scipy(forward_kinematics, perturbed)
                return numerical_result
            else:
                return self.solveIKNumerical_Scipy(forward_kinematics, current_joint_configuration)

        else:
            Q = self.solveIK(forward_kinematics)
            if Q is not None:
                current_joint = np.array(current_joint_configuration)
                delta_Q = np.absolute(Q - current_joint) * self.joint_weights
                closest_ik_index = np.argmin(np.sum(delta_Q, axis=1))
                return Q[closest_ik_index]
            else:
                return None

    def interpolatePose(self, start: np.ndarray, end: np.ndarray, steps: int = 100) -> List[np.ndarray]:
        trajectory = []
        for i in range(steps):
            alpha = i / max((steps - 1), 1)
            pos = (1 - alpha) * start[:3, 3] + alpha * end[:3, 3]
            quat_start = R.from_matrix(start[:3, :3]).as_quat()
            quat_end = R.from_matrix(end[:3, :3]).as_quat()
            quat = (1 - alpha) * quat_start + alpha * quat_end
            quat /= np.linalg.norm(quat)
            rot = R.from_quat(quat).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = pos
            trajectory.append(T)
        return trajectory

    def smoothTrajectory(self, trajectory: List[List[float]], window_size: int = 11, polyorder: int = 3) -> List[List[float]]:
        if window_size % 2 == 0:
            window_size += 1
        traj_array = np.array(trajectory).T
        smoothed = np.zeros_like(traj_array)

        for i in range(6):
            joint_data = traj_array[i]
            if len(joint_data) < window_size:
                smoothed[i] = joint_data
            else:
                smoothed[i] = savgol_filter(joint_data, window_size, polyorder)
        return smoothed.T.tolist()

    def planPath_RRT(
        self,
        waypoints: List[np.ndarray],
        max_iter: int = 2000,
        goal_sample_rate: float = 0.2,
        search_radius: float = 0.3,
        joint_bounds: Optional[List[Tuple[float, float]]] = None,
        start_config: Optional[List[float]] = None,
        smooth: bool = True,
        show_animation: bool = False
    ) -> List[List[float]]:
        if not waypoints:
            print("❌ 无目标位姿")
            return []

        if joint_bounds is None:
            joint_bounds = [[self.limit_min, self.limit_max] for _ in range(6)]

        path = []
        current_start_config = start_config or [math.radians(angle) for angle in [0, -90, 0, -90, 0, 0]]

        for target_pose in waypoints:
            print(f"🎯 规划到目标位姿:\n{target_pose}")
            rrt_result = self._rrt_search(target_pose, current_start_config, joint_bounds, max_iter, goal_sample_rate, search_radius, show_animation)
            if rrt_result is None:
                print("❌ 无法找到路径")
                return []
            path_segment, current_start_config = rrt_result
            path.extend(path_segment)

        if smooth and len(path) > 0:
            print("🌀 平滑轨迹...")
            path = self.smoothTrajectory(path)

        return path

    def _rrt_search(self, target_pose, start_config, joint_bounds, max_iter, goal_sample_rate, search_radius, animation=False):
        from collections import defaultdict
        import random

        class Node:
            def __init__(self, q):
                self.q = np.array(q)
                self.parent = None
                self.cost = 0.0

        nodes = [Node(start_config)]
        best_goal_node = None
        c_best = float('inf')
        start_time = time.time()
        timeout = 10
        no_improvement = 0
        MAX_NO_IMPROVEMENT = 200

        x_center = np.array(start_config)
        goal_config = self.findClosestIK(target_pose, start_config)
        if goal_config is None:
            print("❌ 目标构型不可达")
            return None
        goal_config = np.array(goal_config)
        c_min = np.linalg.norm(goal_config - x_center)
        L = np.eye(6)

        for i in range(max_iter):
            if time.time() - start_time > timeout:
                print("⏰ 超时退出")
                break

            if random.random() < goal_sample_rate:
                rnd = goal_config.copy()
            else:
                if c_best == float('inf'):
                    # 使用 Halton 采样代替随机采样
                    h_sample = self._halton_sequence(i, len(joint_bounds))
                    rnd = []
                    for j, (lb, ub) in enumerate(joint_bounds):
                        rnd.append(lb + h_sample[j] * (ub - lb))
                    rnd = np.array(rnd)
                else:
                    while True:
                        rnd_ellipsoid = self._sample_in_ellipsoid(x_center, c_best, c_min, L, joint_bounds)
                        T = self.fk(rnd_ellipsoid)
                        if self.isPoseReachable(T):
                            break
                    rnd = rnd_ellipsoid

            nearest_ind = min(range(len(nodes)), key=lambda i: np.linalg.norm(nodes[i].q - rnd))
            nearest_node = nodes[nearest_ind]

            dir_vec = rnd - nearest_node.q
            length = np.linalg.norm(dir_vec)
            if length == 0:
                continue
            dir_vec /= length
            new_q = nearest_node.q + dir_vec * min(search_radius, length)

            T = self.fk(new_q.tolist())
            if not self.isPoseReachable(T):
                continue

            new_node = Node(new_q)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + length

            near_inds = self._find_near_nodes(new_node, nodes, radius=search_radius)
            if near_inds:
                new_node = self._choose_parent(new_node, near_inds, nodes)
                if new_node is None:
                    continue
                self._rewire(new_node, near_inds, nodes)

            nodes.append(new_node)

            if self._is_close_to_target(new_node.q, target_pose):
                if new_node.cost < c_best:
                    print(f"🆕 更新最优路径，cost={new_node.cost}")
                    best_goal_node = new_node
                    c_best = new_node.cost
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1

            if no_improvement > MAX_NO_IMPROVEMENT:
                print("🔚 多次无改进，提前终止")
                break

            if animation and i % 50 == 0:
                self._draw_graph(nodes, target_pose, sampled_point=rnd)

        if best_goal_node is None:
            print("⚠️ 达到最大迭代次数仍未找到路径")
            return None

        path = []
        node = best_goal_node
        while node:
            path.append(node.q.tolist())
            node = node.parent
        path.reverse()
        return path, path[-1]

    def _sample_in_ellipsoid(self, center, c_best, c_min, L, bounds):
        dim = len(center)
        while True:
            r = np.random.randn(dim)
            r /= np.linalg.norm(r)
            r *= np.random.rand()
            r *= c_best / 2
            r += center
            valid = all(bounds[i][0] <= r[i] <= bounds[i][1] for i in range(dim))
            if valid:
                return r.tolist()

    def _halton_sequence(self, index: int, dim: int) -> np.ndarray:
        """
        生成第 index 个 Halton 序列点
        :param index: 序列索引（从1开始）
        :param dim: 维度
        :return: Halton 序列点
        """
        primes = [2, 3, 5, 7, 11, 13][:dim]  # 使用前 dim 个质数作为基底
        result = np.zeros(dim)

        for i, prime in enumerate(primes):
            d = index + 1  # Halton 通常从1开始计数
            x = 0.0
            inv_prime = 1.0 / prime
            factor = 1.0

            while d > 0:
                factor *= inv_prime
                d, remainder = divmod(d, prime)
                x += remainder * factor

            result[i] = x

        return result


    def _find_near_nodes(self, new_node, nodes, radius=0.5):
        dist_list = [np.linalg.norm(n.q - new_node.q) for n in nodes]
        return [i for i, d in enumerate(dist_list) if d <= radius]

    def _choose_parent(self, new_node, near_inds, nodes):
        costs = []
        for i in near_inds:
            if self._check_connectivity(nodes[i].q, new_node.q):
                cost = nodes[i].cost + np.linalg.norm(new_node.q - nodes[i].q)
                costs.append((cost, i))
        if not costs:
            return None
        _, parent_idx = min(costs, key=lambda x: x[0])
        new_node.parent = nodes[parent_idx]
        new_node.cost = new_node.parent.cost + np.linalg.norm(new_node.q - new_node.parent.q)
        return new_node

    def _rewire(self, new_node, near_inds, nodes):
        for i in near_inds:
            if self._check_connectivity(new_node.q, nodes[i].q):
                new_cost = new_node.cost + np.linalg.norm(nodes[i].q - new_node.q)
                if new_cost < nodes[i].cost:
                    nodes[i].parent = new_node
                    nodes[i].cost = new_cost

    def _is_close_to_target(self, q, target_pose, threshold=0.01):
        T = self.fk(q)
        pos_error = T[:3, 3] - target_pose[:3, 3]
        orient_error = R.from_matrix(T[:3, :3]).as_rotvec() - R.from_matrix(target_pose[:3, :3]).as_rotvec()
        return np.linalg.norm(pos_error) < threshold and np.linalg.norm(orient_error) < threshold

    def _check_connectivity(self, q1, q2, steps=10):
        key = (tuple(np.round(q1, 3)), tuple(np.round(q2, 3)))
        if hasattr(self, '_connectivity_cache') and key in self._connectivity_cache:
            return self._connectivity_cache[key]

        for alpha in np.linspace(0, 1, steps):
            q = (1 - alpha) * np.array(q1) + alpha * np.array(q2)
            T = self.fk(q.tolist())
            if not self.isPoseReachable(T):
                self._connectivity_cache[key] = False
                return False

        self._connectivity_cache[key] = True
        return True

    def _draw_graph(self, nodes, target_pose, sampled_point=None):
        plt.cla()
        xs = [node.q[0] for node in nodes]
        ys = [node.q[1] for node in nodes]
        plt.plot(xs, ys, '.', markersize=2)

        if sampled_point is not None:
            plt.plot(sampled_point[0], sampled_point[1], 'go', label='Sampled Point')

        plt.plot(target_pose[0, 3], target_pose[1, 3], "xr", label='Target')
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)