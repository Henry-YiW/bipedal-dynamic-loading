
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
unitree_mujoco_test_dir = os.path.join(current_dir, "..")
sys.path.append(unitree_mujoco_test_dir)

import torch
import torch.nn as nn
from module.modules import Actor, MLPEncoder

original_model_path = '/home/hcg/CLX/projects/legged_load_adaptation/experiment/ablation_experiment/target_model_c_for_experiment/model_c_quad_rl_load/legged_gym/logs/go2_load_teacher_student_phase_model_c/Sep22_12-30-52_reinforce/model_34250.pt'
extracted_model_save_path = '/home/hcg/CLX/projects/legged_load_adaptation/quadruped_rl_load_mujoco_test/unitree_mujoco_test/model/blind_locomotion_load'

# 加载源模型
model_path = original_model_path
model_state_dict = torch.load(model_path, weights_only=True)
print(model_state_dict['model_state_dict'].keys())

# 提取 actor
actor_state_dict = {
    k.replace('actor.','actor.'): v for k, v in model_state_dict['model_state_dict'].items() if k.startswith('actor.')}
print(actor_state_dict.keys())
actor = Actor(num_obs=77+8, num_actions=12)
actor.load_state_dict(actor_state_dict)
torch.save(actor.state_dict(), extracted_model_save_path + '/actor/actor_0922_3.pth')

# 提取 proprioceptive_encoder
proprioceptive_encoder = MLPEncoder(input_dim=45*15, hidden_dims=[512, 256, 128], latent_dim=32, activation='elu')
encoder_state_dict = {
    k.replace('proprioceptive_encoder.',''): v for k, v in model_state_dict['model_state_dict'].items() if k.startswith('proprioceptive_encoder.')}
print(encoder_state_dict.keys())
proprioceptive_encoder.load_state_dict(encoder_state_dict)
torch.save(proprioceptive_encoder.state_dict(), extracted_model_save_path + '/proprio_encoder/proprio_0922_3.pth')

# # 提取 load state estimator
load_state_estimator = MLPEncoder(input_dim=45*15, hidden_dims=[512, 256, 64], latent_dim=8, activation='elu')
encoder_state_dict = {
    k.replace('load_state_estimator.',''): v for k, v in model_state_dict['model_state_dict'].items() if k.startswith('load_state_estimator.')}
print(encoder_state_dict.keys())
load_state_estimator.load_state_dict(encoder_state_dict)
torch.save(load_state_estimator.state_dict(), extracted_model_save_path + '/load_state_estimator/load_estimator_0922_3.pth')

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
