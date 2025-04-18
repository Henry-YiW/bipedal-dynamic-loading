#!/usr/bin/env python3

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)
actor_dir = os.path.join(unitree_mujoco_test_dir, "model/blind_locomotion/actor")
proprio_encoder_dir = os.path.join(unitree_mujoco_test_dir, "model/blind_locomotion/proprio_encoder")
robot_data_dir = os.path.join(unitree_mujoco_test_dir, "data")

import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from module.modules import Actor, MLPEncoder
from pynput import keyboard


def quat_rotate_inverse(q, v):
    # 假设 q 的形状为 (4,)，v 的形状为 (3,)
    q_w = q[0]
    q_vec = q[1:]
    # 计算部分 a
    a = v * (2.0 * q_w**2 - 1.0)
    # 计算部分 b
    b = np.cross(q_vec, v) * q_w * 2.0
    # 计算部分 c
    c = q_vec * (np.dot(q_vec, v) * 2.0)

    return a - b + c


##############
# 初始化键盘监听
##############
# 全局变量，保存键盘状态
key_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "l": False,
    "r": False,
}
# 最大和最小速度限制
max_speed = 1.0
min_speed = -1.0
# 阻尼系数，用于速度平滑下降
damping_factor = 0.2  # 调整该值以改变速度衰减快慢


def on_press(key):
    """按键按下时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = True
    elif key == keyboard.Key.down:
        key_state["down"] = True
    elif key == keyboard.Key.left:
        key_state["left"] = True
    elif key == keyboard.Key.right:
        key_state["right"] = True
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = True
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = True


def on_release(key):
    """按键释放时的回调函数"""
    if key == keyboard.Key.up:
        key_state["up"] = False
    elif key == keyboard.Key.down:
        key_state["down"] = False
    elif key == keyboard.Key.left:
        key_state["left"] = False
    elif key == keyboard.Key.right:
        key_state["right"] = False
    elif key == keyboard.KeyCode.from_char("r"):
        key_state["l"] = False
    elif key == keyboard.KeyCode.from_char("l"):
        key_state["r"] = False


##############
# 加载机器人模型
##############
model = mujoco.MjModel.from_xml_path(robot_data_dir + "/go2/scene_terrain.xml")
data = mujoco.MjData(model)

############################
# 加载 policy 和 encoder 模型
############################
device = torch.device("cuda")
actor = Actor(num_obs=77, num_actions=12, hidden_dims=[512, 256, 128])
# actor.load_state_dict(torch.load(actor_dir + "/actor_0910_cosine_reinforce.pth"))
actor.load_state_dict(torch.load(actor_dir + "/actor_oracle.pth"))
actor = actor.to(device)
actor.eval()
proprio_encoder = MLPEncoder(input_dim=45*15)
# proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_encoder_0910_cosine_reinforce.pth"))
proprio_encoder.load_state_dict(torch.load(proprio_encoder_dir + "/proprio_oracle.pth"))
proprio_encoder = proprio_encoder.to(device)
proprio_encoder.eval()

###################
# 初始化储存状态的变量
###################
global command
body_pos = np.zeros(3)  # 机器人的位置 在世界坐标系下
body_quat = np.zeros(4)  # 机器人的orientation 在世界坐标系下
body_lin_vel = np.zeros(3)
body_ang_vel = np.zeros(3)
gravity_projection = np.zeros(3)
command = np.zeros(3)
joint_pos = np.zeros(12)
joint_vel = np.zeros(12)
last_action = np.zeros(12)
torques = np.zeros(12)
default_joint_angles = {  # = target angles [rad] when action = 0.0
    "FL_hip_joint": 0.1,  # [rad]
    "FL_thigh_joint": 0.8,  # [rad]
    "FL_calf_joint": -1.5,  # [rad]

    "FR_hip_joint": -0.1,  # [rad]
    "FR_thigh_joint": 0.8,  # [rad]
    "FR_calf_joint": -1.5,  # [rad]

    "RL_hip_joint": 0.1,  # [rad]
    "RL_thigh_joint": 1.0,  # [rad]
    "RL_calf_joint": -1.5,  # [rad]

    "RR_hip_joint": -0.1,  # [rad]
    "RR_thigh_joint": 1.0,  # [rad]
    "RR_calf_joint": -1.5,  # [rad]
}
default_dof_pos = np.array(list(default_joint_angles.values()))

#####################################
# 力矩计算相关增益以及 observation scale
#####################################
p_gains = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])  # 位置项增益
d_gains = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # 速度项增益
torque_limits = np.array([20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0, 20.0, 55.0, 55.0])
actions_scale = 0.25
body_lin_vel_scale = 2.0
body_ang_vel_scale = 0.25
command_scale = np.array([2.0, 2.0, 0.25])
joint_vel_scale = 0.05
joint_pos_scale = 1.0

##############
# 关节状态初始化
##############
data.qpos[7:19] = default_dof_pos

#########################
# 监听键盘输入以发送 command
#########################
command = np.array([0.0, 0.0, 0.0])
prev_command = np.array([0.0, 0.0, 0.0])
print(
    "Use arrow keys to move the robot.\n",
    "'l': turn left\n",
    "'r': turn left\n",
    "UP: move forward,\n",
    "DOWN: move backward,\n",
    "LEFT: move left,\n" "RIGHT: move right.\n",
)
# 监听键盘输入
command_scale_factor = 0.005
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
proprio_obs_history = torch.zeros((1, 45*15)).to(device)

###########
# 仿真主循环
###########
m = model
d = data

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [-7., 4.5, 3.]  # Example: move camera to (0, 0, 0.5)
    viewer.cam.azimuth = 20 # Example: rotate camera by 180 degree
    viewer.cam.elevation = -30  # Example: tilt camera by 30 degree
    start = time.time()
    last_update_time = start
    # max_command_value = 0.5
    while viewer.is_running() and time.time() - start < 5000:
        # 持续更新 command，按住键时加速或减速
        if key_state["up"]: command[0] = min(command[0] + command_scale_factor, max_speed)
        elif key_state["down"]: command[0] = max(command[0] - command_scale_factor, min_speed)
        else:
            # 平滑衰减 x 方向速度
            if command[0] > 0: command[0] = max(command[0] - damping_factor, 0)
            elif command[0] < 0: command[0] = min(command[0] + damping_factor, 0)

        if key_state["left"]: command[1] = min(command[1] + command_scale_factor, max_speed)
        elif key_state["right"]: command[1] = max(command[1] - command_scale_factor, min_speed)
        else:
            # 平滑衰减 y 方向速度
            if command[1] > 0: command[1] = max(command[1] - damping_factor, 0)
            elif command[1] < 0: command[1] = min(command[1] + damping_factor, 0)

        if key_state["l"]: command[2] = max(command[2] - command_scale_factor, -1.0)  # 限制角速度最小为 -1
        elif key_state["r"]: command[2] = min(command[2] + command_scale_factor, 1.0)  # 限制角速度最大为 1
        else:
            # 平滑衰减角速度
            if command[2] > 0: command[2] = max(command[2] - damping_factor, 0)
            elif command[2] < 0: command[2] = min(command[2] + damping_factor, 0)

        ############
        # 更新模型输入
        ############
        current_time = time.time()

        # body_lin_vel = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array(data.qvel[0:3]))
        body_ang_vel = np.array(data.qvel[3:6])  # mujoco 中的角速度是在局部坐标系下的，所以不需要转换
        gravity_projection = quat_rotate_inverse(np.array(data.qpos[3:7]), np.array([0, 0, -1.0]))
        command_input = command
        joint_pos = np.array(data.qpos[7:19])
        joint_vel = np.array(data.qvel[6:18])

        proprio_observation = np.concatenate(
            [
                body_ang_vel * body_ang_vel_scale,
                gravity_projection,
                command_input * command_scale,
                (joint_pos - default_dof_pos) * joint_pos_scale,
                joint_vel * joint_vel_scale,
                last_action,
            ]
        )

        # noise_strength = 0.02  # 噪声的强度，值越大，噪声越强
        # noise = np.random.randn(*proprio_observation.shape) * noise_strength  # 生成与 proprio_observation 形状相同的噪声
        # proprio_observation += noise  # 将噪声添加到 proprio_observation# 加入噪声

        proprio_observation = torch.from_numpy(proprio_observation).float().to(device).unsqueeze(0) # shape (1,45)
        proprio_obs_history = torch.cat((proprio_obs_history[:,45:], proprio_observation),dim=-1) # shape (1,675)

        #####################################
        # 将 observation 输入模型得到输出 action
        #####################################
        proprio_latent = proprio_encoder(proprio_obs_history)
        proprio_latent = torch.nn.functional.normalize(proprio_latent, p=2, dim=-1)
        actor_input = torch.cat((proprio_observation, proprio_latent), dim=-1)
        
        action = actor(actor_input)
        action = action.detach().cpu().numpy()
        action = np.clip(action, -6.0, 6.0)
        last_action[:] = action[:]

        ####################################
        # 将 action 映射为 torque 作为控制输出
        ####################################
        decimation = 4
        for i in range(decimation):
            # 计算力矩
            dof_pos = np.array(data.qpos[7:19])
            dof_vel = np.array(data.qvel[6:18])
            torques = p_gains * (action * actions_scale + default_dof_pos - dof_pos) - d_gains * dof_vel
            torques = np.clip(torques, -torque_limits, torque_limits)
            data.ctrl[0:12] = torques
            # 执行仿真
            mujoco.mj_kinematics(m, d)
            mujoco.mj_step(model, data)
            time.sleep(0.003)

        ###################
        # 实时输出command命令
        ###################
        if not np.array_equal(command, prev_command):
            print(
                "Use arrow keys to move the robot.\n",
                "'l': turn left\n",
                "'r': turn left\n",
                "UP: move forward,\n",
                "DOWN: move backward,\n",
                "LEFT: move left,\n" "RIGHT: move right.\n",
            )
            print(
                "x_velocity: {:.2f},\ny_velocity: {:.2f},\nangular_velocity: {:.2f}\n".format(
                    command[0], command[1], command[2]
                )
            )
            prev_command = command.copy()

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        viewer.sync()
