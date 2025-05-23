from inverseKinematicsUR5 import InverseKinematicsUR5, transformRobotParameter
import numpy as np

# 示例目标位姿
theta_target = [0.7, 0.3, 0.2, 1.0, 0.5, 0.1]
gd = transformRobotParameter(theta_target)

# 初始化求解器
ik = InverseKinematicsUR5()
ik.enableDebugMode(True)

# 使用 Scipy 数值解法
theta_sol_scipy = ik.findClosestIK(gd, [0.6, 0.2, 0.5, 1.2, 0.1, -0.1], use_numerical=True)
print("Scipy Numerical Solution:", theta_sol_scipy)

# （可选）使用解析解
# theta_sol_analytical = ik.findClosestIK(gd, [0.6, 0.2, 0.5, 1.2, 0.1, -0.1], use_numerical=False)
# print("Analytical Solution:", theta_sol_analytical)