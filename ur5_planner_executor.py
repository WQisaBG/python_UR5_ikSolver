import math
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import time

# 导入已有模块
from inverseKinematicsUR5 import InverseKinematicsUR5


class UR5PlannerExecutor:
    def __init__(self):
        self.ik_solver = InverseKinematicsUR5()
        self.ik_solver.enableDebugMode(False)
        self.joint_trajectory: List[List[float]] = []

    # 设置默认构型
    def setDefaultConfiguration(self, angles_deg: List[float]) -> None:
        """
        设置默认初始构型（角度单位：度）
        """
        self.default_config = [math.radians(a) for a in angles_deg]

    # 设置目标位姿列表
    def setWaypoints(self, waypoints: List[np.ndarray]) -> None:
        """
        设置末端执行器的目标位姿列表
        """
        self.waypoints = waypoints

    # 规划路径
    def planPath(self, steps_per_segment: int = 100, use_numerical: bool = True) -> bool:
        print("🔄 开始路径规划...")
        trajectory = self.ik_solver.planPath(
            self.waypoints,
            steps_per_segment=steps_per_segment,
            use_numerical=use_numerical,
            current_joint_angles=self.default_config,
            perturb_if_singular=True,
            interpolation_method='slerp'
        )
        if not trajectory:
            print("❌ 路径规划失败")
            return False
        print(f"✅ 路径规划完成，共 {len(trajectory)} 步")
        self.joint_trajectory = trajectory
        return True

    # 平滑轨迹
    def smoothTrajectory(self, window_size: int = 11, polyorder: int = 3) -> None:
        print("🌀 正在平滑轨迹...")
        self.joint_trajectory = self.ik_solver.smoothTrajectory(self.joint_trajectory, window_size, polyorder)
        print("✅ 轨迹平滑完成")

    # 绘图显示
    def plotTrajectory(self) -> None:
        print("📈 绘制关节轨迹...")
        self.ik_solver.plotJointTrajectory(self.joint_trajectory)

    # 仿真播放（PyBullet 示例）
    def simulateInPyBullet(self, ur5_id, physicsClient, dt: float = 0.01) -> None:
        """
        使用 PyBullet 播放 UR5 关节轨迹
        参数:
            ur5_id: UR5 机器人的 PyBullet ID
            physicsClient: PyBullet 物理客户端
            dt: 时间步长 (秒)
        """
        try:
            import pybullet as p
        except ImportError:
            print("⚠️ PyBullet 未安装，无法播放")
            return

        num_joints = p.getNumJoints(ur5_id)
        joint_indices = list(range(num_joints))

        print("🎮 开始播放轨迹...")
        for i, q in enumerate(self.joint_trajectory):
            for idx, angle in enumerate(q):
                p.setJointMotorControl2(
                    bodyIndex=ur5_id,
                    jointIndex=idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=500
                )
            p.stepSimulation()
            time.sleep(dt)
        print("🏁 播放完成")

    # 导出为 CSV 文件
    def exportToCSV(self, filename: str = "ur5_trajectory.csv") -> None:
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t'] + [f'q{i+1}' for i in range(6)])
            for i, q in enumerate(self.joint_trajectory):
                writer.writerow([i * 0.01] + q)
        print(f"📁 轨迹已导出至 {filename}")

    # 导出为 ROS JointTrajectory 消息（仅示例结构）
    def exportToROSJointTrajectory(self, topic_name: str = "/joint_path") -> None:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
                                "elbow_joint", "wrist_1_joint",
                                "wrist_2_joint", "wrist_3_joint"]
        for i, q in enumerate(self.joint_trajectory):
            point = JointTrajectoryPoint()
            point.positions = q
            point.time_from_start.sec = int(i * 0.01)
            point.time_from_start.nanosec = int((i * 0.01 - int(i * 0.01)) * 1e9)
            traj_msg.points.append(point)
        print("📤 已生成 ROS JointTrajectory 消息，可通过 rostopic 发布或写入 bag 文件")

    # 发送轨迹到真实 UR5（需连接 ROS）
    def executeOnRealRobot(self, robot_ip: str = "192.168.1.102") -> None:
        import rtde_control
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        print("📡 连接到真实 UR5 机器人...")

        print("🤖 发送轨迹...")
        for i, q in enumerate(self.joint_trajectory):
            rtde_c.moveJ(q, 0.5, 0.5)
        rtde_c.stopScript()
        print("✅ 轨迹执行完成")

    # 主流程示例
    def runDemo(self):
        # 示例目标位姿
        from scipy.spatial.transform import Rotation as R

        pos1 = np.array([0.4, -0.3, 0.4])
        quat1 = R.from_euler('xyz', [0, -np.pi/2, 0]).as_quat()
        T1 = np.eye(4)
        T1[:3, :3] = R.from_quat(quat1).as_matrix()
        T1[:3, 3] = pos1

        pos2 = np.array([0.4, 0.3, 0.4])
        quat2 = R.from_euler('xyz', [0, -np.pi/2, np.pi]).as_quat()
        T2 = np.eye(4)
        T2[:3, :3] = R.from_quat(quat2).as_matrix()
        T2[:3, 3] = pos2

        self.setDefaultConfiguration([0, -90, 0, -90, 0, 0])  # 默认姿态
        self.setWaypoints([T1, T2])

        success = self.planPath(steps_per_segment=100, use_numerical=True)
        if not success:
            print("❌ 路径规划失败")
            return

        self.smoothTrajectory(window_size=11, polyorder=3)
        self.plotTrajectory()

        # 如果需要，取消注释下面行使用 PyBullet 播放
        # self.simulateInPyBullet(ur5_id, physicsClient)

        # 如果需要，导出轨迹
        # self.exportToCSV()

        # 如果需要，发送到真实机器人
        # self.executeOnRealRobot()


if __name__ == "__main__":
    planner = UR5PlannerExecutor()
    planner.runDemo()