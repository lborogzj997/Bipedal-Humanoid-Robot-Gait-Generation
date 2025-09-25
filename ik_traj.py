import pybullet as p
import pybullet_data
import numpy as np
import time
import csv
from scipy.spatial.transform import Rotation as R

use_gui = True

def solve_dual_foot_ik(robot_id, right_foot_name, left_foot_name,
                       target_pos_right, target_orn_right,
                       target_pos_left, target_orn_left,
                       max_iters=10, threshold=1e-8):
    right_index = -1
    left_index = -1
    num_joints = p.getNumJoints(robot_id)

    joint_indices = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []

    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        link_name = joint_info[12].decode("utf-8")

        if link_name == right_foot_name:
            right_index = i
        elif link_name == left_foot_name:
            left_index = i

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_indices.append(i)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
            joint_ranges.append(joint_info[9] - joint_info[8])
            rest_poses.append(p.getJointState(robot_id, i)[0])

    name_to_index = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in joint_indices}

    for knee_joint_name in ['left_knee_joint', 'right_knee_joint']:
        if knee_joint_name in name_to_index:
            idx = joint_indices.index(name_to_index[knee_joint_name])
            rest_poses[idx] = 1.5

    if right_index == -1 or left_index == -1:
        raise ValueError("Foot link names not found.")

    joint_poses_right = p.calculateInverseKinematics(
        robot_id,
        right_index,
        target_pos_right,
        target_orn_right,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        jointDamping=[0.1] * len(joint_indices),
        maxNumIterations=max_iters,
        residualThreshold=threshold
    )

    for idx, joint_index in enumerate(joint_indices):
        p.resetJointState(robot_id, joint_index, joint_poses_right[joint_index])

    joint_poses_left = p.calculateInverseKinematics(
        robot_id,
        left_index,
        target_pos_left,
        target_orn_left,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        jointDamping=[0.1] * len(joint_indices),
        maxNumIterations=max_iters,
        residualThreshold=threshold
    )

    return joint_poses_left


# PyBullet setup
if use_gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

urdf_path = r"sz13\urdf\sz13.urdf"
robot_id = p.loadURDF(urdf_path, useFixedBase=False)

# Get waist joint indices
waist_pitch_index = [i for i in range(p.getNumJoints(robot_id))
                     if p.getJointInfo(robot_id, i)[1].decode('utf-8') == 'waist_pitch_joint'][0]
waist_roll_index = [i for i in range(p.getNumJoints(robot_id))
                    if p.getJointInfo(robot_id, i)[1].decode('utf-8') == 'waist_roll_joint'][0]
waist_yaw_index = [i for i in range(p.getNumJoints(robot_id))
                   if p.getJointInfo(robot_id, i)[1].decode('utf-8') == 'waist_yaw_joint'][0]

# Load trajectory data
csv_path = r"foot_trajectory.csv"
frames = []
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        if len(row) < 11:
            continue
        frames.append([float(val) for val in row[1:]])

# Output setup
output_path = "joint_positions_output.csv"
output_file = open(output_path, 'w')
frame_idx = 1

alpha = 0.2
filtered_joint_positions = None

# Camera control
camera_distance = 1.5
camera_pitch = -20
camera_yaw = 50
camera_target = [0, 0, 0]

def handle_camera_keys():
    global camera_yaw, camera_pitch
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        camera_yaw -= 1
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        camera_yaw += 1
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        camera_pitch += 1
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        camera_pitch -= 1

    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=camera_target
    )

for frame in frames:
    target_pos_right = frame[0:3]
    pitch_right_deg = frame[3]
    pitch_right_rad = np.radians(pitch_right_deg)
    target_orn_right = p.getQuaternionFromEuler([0, pitch_right_rad, 0])

    target_pos_left = frame[4:7]
    pitch_left_deg = frame[7]
    pitch_left_rad = np.radians(pitch_left_deg)
    target_orn_left = p.getQuaternionFromEuler([0, pitch_left_rad, 0])

    base_link_pos = frame[8:11]
    roll_base = frame[11]  # roll in radians

    base_orn = p.getQuaternionFromEuler([roll_base, 0, 0])
    p.resetBasePositionAndOrientation(robot_id, base_link_pos, base_orn)

    # Compute waist joint angles to cancel base_link orientation
    base_inv_quat = p.invertTransform([0, 0, 0], base_orn)[1]
    r = R.from_quat(base_inv_quat)
    waist_pitch, waist_roll, waist_yaw = r.as_euler('yxz')

    # Apply waist joints in simulation
    p.resetJointState(robot_id, waist_pitch_index, waist_pitch)
    p.resetJointState(robot_id, waist_roll_index, waist_roll)
    p.resetJointState(robot_id, waist_yaw_index, waist_yaw)

    # Solve IK for the feet
    joint_positions = solve_dual_foot_ik(
        robot_id,
        "right_foot_link",
        "left_foot_link",
        target_pos_right,
        target_orn_right,
        target_pos_left,
        target_orn_left
    )

    if filtered_joint_positions is None:
        filtered_joint_positions = joint_positions
    else:
        filtered_joint_positions = [
            alpha * new + (1 - alpha) * old
            for new, old in zip(joint_positions, filtered_joint_positions)
        ]

    # Output: frame index, joint positions, foot pitches, waist orientation
    line = f"{frame_idx} " + ' '.join(f'{jp:.6f}' for jp in filtered_joint_positions)
    output_file.write(line + '\n')
    frame_idx += 1

    # Apply filtered joint positions (excluding waist, already applied)
    for i in range(p.getNumJoints(robot_id)):
        joint_name = p.getJointInfo(robot_id, i)[1].decode('utf-8')
        joint_type = p.getJointInfo(robot_id, i)[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            if joint_name not in ['waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint']:
                p.resetJointState(robot_id, i, filtered_joint_positions[i])

    handle_camera_keys()

    if use_gui:
        time.sleep(1.0 / 240.0)

output_file.close()
p.disconnect()

from scipy.signal import savgol_filter

# 假设 joint_positions_output.csv 结果已经写入
# 读取 back output，然后全局 filter

import numpy as np

# 重新读取 output file
data = np.loadtxt('joint_positions_output.csv')

frame_idx = data[:, 0]
joint_data = data[:, 1:]

# 对每个关节序列单独平滑
window_size = 9  # 必须是奇数，可调
poly_order = 2

for j in range(joint_data.shape[1]):
    joint_data[:, j] = savgol_filter(joint_data[:, j], window_size, poly_order)

# 重写覆盖 output file（或另存）
with open('joint_positions_output_filtered.csv', 'w') as f:
    for idx, row in zip(frame_idx, joint_data):
        line = f"{int(idx)} " + ' '.join(f"{v:.6f}" for v in row)
        f.write(line + '\n')