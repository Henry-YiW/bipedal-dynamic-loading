# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=32, activation='elu'):
        super(Encoder, self).__init__()
        activation = get_activation(activation)
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                encoder_layers.append(nn.Linear(hidden_dims[l], latent_dim))
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
    
    def forward(self, x):
        return self.encoder(x)
    
    
class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        obs_history_length=15,
                        # actor_hidden_dims=[256, 256, 256],
                        # critic_hidden_dims=[256, 256, 256],
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        encoder_hidden_dims=[512, 256, 128],
                        estimator_hidden_dims=[512, 256, 64],
                        encoder_latent_dim=32, # encoder latent vector 维度
                        load_state_dim=8,
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs + encoder_latent_dim + load_state_dim # proprio + encoded latent 
        mlp_input_dim_c = num_critic_obs + encoder_latent_dim + load_state_dim

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Proprioceptive Encoder
        self.proprioceptive_encoder = Encoder(input_dim=num_actor_obs*obs_history_length, hidden_dims=encoder_hidden_dims,
                                                    latent_dim=encoder_latent_dim, activation="elu")
        # Privileged Encoder
        self.privileged_encoder = Encoder(input_dim=num_critic_obs, hidden_dims=encoder_hidden_dims,
                                                    latent_dim=encoder_latent_dim, activation="elu")
        # Load state estimator
        self.load_state_estimator = Encoder(input_dim=num_actor_obs*obs_history_length, hidden_dims=estimator_hidden_dims,
                                                    latent_dim=load_state_dim, activation="elu")
    
        print(f"Privileged Encoder MLP: {self.privileged_encoder}")
        print(f"Proprioceptive Encoder MLP: {self.proprioceptive_encoder}")
        print(f"Load State Estimator MLP: {self.load_state_estimator}")


        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # def update_distribution(self, observations):
    #     mean = self.actor(observations)
    #     self.distribution = Normal(mean, mean*0. + self.std)

    def update_distribution(self, observations, observations_history, critic_observations):
        estimated_load_state = self.load_state_estimator(observations_history)
        # latent = self.privileged_encoder(critic_observations)
        latent = self.proprioceptive_encoder(observations_history)
        latent = nn.functional.normalize(latent, p=2, dim=-1)
        combined_observations = torch.cat((observations, latent, estimated_load_state), dim=1)
        # print(f"Combined observations shape: {combined_observations.shape}")
        # print(f"Observations shape: {observations.shape}")
        # print(f"Latent shape: {latent.shape}")
        # print(f"Estimated load state shape: {estimated_load_state.shape}")
        mean = self.actor(combined_observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    # def act(self, observations, **kwargs):
    #     self.update_distribution(observations)
    #     return self.distribution.sample()

    def act(self, observations, observations_history, critic_observations, **kwargs):
        self.update_distribution(observations, observations_history, critic_observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_inference(self, observations):
    #     actions_mean = self.actor(observations)
    #     return actions_mean
    
    def act_inference(self, observations, observations_history, critic_observations):
        latent = self.proprioceptive_encoder(observations_history)
        estimated_load_state = self.load_state_estimator(observations_history)
        # latent = self.privileged_encoder(critic_observations)
        latent = nn.functional.normalize(latent, p=2, dim=-1)
        actions_mean = self.actor(torch.cat((observations, latent, estimated_load_state), dim=1))
        return actions_mean

    # def evaluate(self, critic_observations, **kwargs):
    #     value = self.critic(critic_observations)
    #     return value
    
    def evaluate(self, critic_observations, load_observations, **kwargs):
        latent = self.privileged_encoder(critic_observations)
        latent = nn.functional.normalize(latent, p=2, dim=-1)
        value = self.critic(torch.cat((critic_observations, latent, load_observations), dim=1))
        return value

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
