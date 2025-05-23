import math
import numpy as np
from typing import List, Optional, Tuple, Union
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

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

    # 优化目标函数：位置 + 姿态误差
    def objective_func(self, theta: np.ndarray, target_pose: np.ndarray) -> float:
        T = self.fk(theta)
        pos_error = T[:3, 3] - target_pose[:3, 3]
        orient_error = R.from_matrix(T[:3, :3]).as_rotvec() - R.from_matrix(target_pose[:3, :3]).as_rotvec()
        weight_pos = 1.0
        weight_orient = 0.5
        return weight_pos * np.linalg.norm(pos_error) + weight_orient * np.linalg.norm(orient_error)

    # 使用 Scipy 进行非线性优化的数值解法
    def solveIKNumerical_Scipy(
        self,
        forward_kinematics: np.ndarray,
        current_joint_angles: List[float],
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Optional[np.ndarray]:
        x0 = np.array(current_joint_angles).copy()
        bounds = [(self.limit_min, self.limit_max) for _ in range(6)]

        result = minimize(
            fun=lambda x: self.objective_func(x, forward_kinematics),
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            tol=tol,
            options={'maxiter': max_iter, 'disp': False}
        )

        if result.success:
            print(f"✅ 成功收敛于 {result.nit} 次迭代")
            return result.x
        else:
            print("⚠️ 未成功收敛")
            print("最终误差:", result.fun)
            return None

    # 找到最接近当前构型的 IK 解
    def findClosestIK(self, forward_kinematics: np.ndarray, current_joint_configuration: List[float], use_numerical: bool = False) -> Optional[np.ndarray]:
        if use_numerical:
            return self.solveIKNumerical_Scipy(forward_kinematics, current_joint_configuration)
        else:
            Q = self.solveIK(forward_kinematics)
            if Q is not None:
                current_joint = np.array(current_joint_configuration)
                delta_Q = np.absolute(Q - current_joint) * self.joint_weights
                delta_Q_weights = np.sum(delta_Q, axis=1)
                closest_ik_index = np.argmin(delta_Q_weights)

                if self.debug:
                    print(f'delta_Q weights for each solutions: {delta_Q_weights}')
                    print(f'Closest IK solution: {Q[closest_ik_index,:]}')

                return Q[closest_ik_index, :]
            else:
                return None