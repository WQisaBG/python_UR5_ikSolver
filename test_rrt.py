from inverseKinematicsUR5 import InverseKinematicsUR5
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


ik = InverseKinematicsUR5()
ik.enableDebugMode(False)

# 设置两个目标位姿
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

waypoints = [T1, T2]

def is_pose_reachable(ik, T):
    ik_result = ik.solveIK(T)
    if ik_result is not None and len(ik_result) > 0:
        print("✅ 解析解存在")
        return True
    else:
        print("❌ 解析解不存在")
        return False

is_pose_reachable(ik, waypoints[0])
is_pose_reachable(ik, waypoints[1])

# 使用 Informed-RRT* 规划路径
joint_path = ik.planPath_RRT(
    waypoints,
    max_iter=5000,
    goal_sample_rate=0.2,
    search_radius=0.3,
    start_config=[math.radians(a) for a in [45, -60, 30, -120, 0, 0]],
    smooth=True,
    show_animation=True
)

if joint_path:
    print("✅ 成功找到路径！")
    print(f"总步数: {len(joint_path)}")
    ik.plotJointTrajectory(joint_path)
else:
    print("❌ 路径规划失败")