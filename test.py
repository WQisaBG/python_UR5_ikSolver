from inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import numpy as np
import math

# 示例目标位姿
theta1 = [0.7, 0.3, 0.2, 1.0, 0.5, 0.1]
theta2 = [1.0, 0.5, 0.0, 0.8, 0.3, 0.0]
theta3 = [1.5, 0.1, -0.2, 1.2, -0.2, 0.0]

gd1 = transformRobotParameter(theta1)
gd2 = transformRobotParameter(theta2)
gd3 = transformRobotParameter(theta3)

# 初始化求解器
ik = InverseKinematicsUR5()
ik.enableDebugMode(True)

# 获取一个非奇异初始构型
initial_guess = ik.findClosestIK(gd1, [math.radians(a) for a in [0, -90, 0, -90, 0, 0]], use_numerical=False)
if initial_guess is None:
    initial_guess = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
else:
    initial_guess = initial_guess.tolist()

# 设置路径点
waypoints = [gd1, gd2, gd3]


# 路径规划
joint_path = ik.planPath(
    waypoints,
    steps_per_segment=10,
    use_numerical=True,
    current_joint_angles=initial_guess,
    perturb_if_singular=True,
    interpolation_method='slerp'  # 更自然的姿态过渡
    )
if joint_path:
    print("✅ 成功生成路径！")
    print(f"总步数: {len(joint_path)}")

    # 平滑处理
    smoothed_path = ik.smoothTrajectory(joint_path, window_size=5)

    # 绘图
    ik.plotJointTrajectory(smoothed_path)
else:
    print("❌ 路径规划失败")