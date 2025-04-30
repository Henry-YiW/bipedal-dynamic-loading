import os
import sys
from typing import Dict

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, quat_rotate_inverse, quat_apply
from legged_gym.utils.terrain import Terrain
from .pointfoot import PointFoot
from legged_gym.envs.pointfoot.PF.pointfoot_with_load_config import PointFootWithLoadCfg

# Helper function from parent, needed if not inheriting __init__ fully
def get_axis_params(value, axis_idx, x_value=0., y_value=0., z_value=0., device='cpu'):
    """ Creates a tensor with requested number of elements, padded with zeros except for the designated axis index.
        \[ explicit axis indices: 0=x, 1=y, 2=z \]
    Args:
        value (float): value to put in the designated axis
        axis_idx (int): index of the axis to use (0, 1 or 2)
        x_value (float): defaults to 0, value to put in x axis index
        y_value (float): defaults to 0, value to put in y axis index
        z_value (float): defaults to 0, value to put in z axis index
        device (str): device to create tensor on
    Returns:
        (torch.Tensor): tensor of shape (3,)
    """
    if axis_idx == 0:
        x_value = value
    elif axis_idx == 1:
        y_value = value
    elif axis_idx == 2:
        z_value = value
    return torch.tensor([x_value, y_value, z_value], device=device, dtype=torch.float)

class PointFootWithLoad(PointFoot):
    cfg = PointFootWithLoadCfg()
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Initialize PointFoot with load functionality. """
        self.has_container = cfg.load.use_container
        self.num_actors_per_env = 2 if not self.has_container else 3  # Robot + Load (+ Container)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        """ Initialize additional buffers for load handling """
        super()._init_buffers()
        
        # Initialize load-specific buffers
        self.load_states = self.root_states[:, 1].clone()  # Shape: (num_envs, 13)
        self.load_relative_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.load_relative_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Initialize container buffers if enabled
        if self.has_container:
            self.container_states = self.root_states[:, 2].clone()  # Shape: (num_envs, 13)
            self.container_relative_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.container_relative_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def compute_observations(self):
        """ Add load observations to the basic robot observations """
        super().compute_observations()
        
        # Add load observations
        load_obs = torch.cat((
            self.load_relative_position * self.obs_scales.lin_vel,
            self.load_relative_velocity * self.obs_scales.lin_vel
        ), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf, load_obs), dim=-1)
        
        # Add container observations if enabled
        if self.has_container:
            container_obs = torch.cat((
                self.container_relative_position * self.obs_scales.lin_vel,
                self.container_relative_velocity * self.obs_scales.lin_vel
            ), dim=-1)
            self.obs_buf = torch.cat((self.obs_buf, container_obs), dim=-1)
        
        return self.obs_buf

    def post_physics_step(self):
        """ Add load state updates to the basic robot state updates """
        super().post_physics_step()

        # Update load state
        self.load_states = self.root_states[:, 1].clone()
        self.load_pos = self.load_states[:, :3]
        self.load_vel = self.load_states[:, 7:10]
        self.load_quat = self.load_states[:, 3:7]
        self.load_ang_vel = self.load_states[:, 10:13]

        # Update relative positions and velocities in robot frame
        base_pos = self.root_states[:, 0, :3]
        base_rot = self.root_states[:, 0, 3:7]
        base_vel = self.root_states[:, 0, 7:10]

        # Compute load relative states in robot frame
        self.load_relative_position = quat_rotate_inverse(base_rot, self.load_pos - base_pos)
        self.load_relative_velocity = quat_rotate_inverse(base_rot, self.load_vel - base_vel)

        # Update container state if enabled
        if self.has_container:
            self.container_states = self.root_states[:, 2].clone()
            self.container_pos = self.container_states[:, :3]
            self.container_vel = self.container_states[:, 7:10]
            self.container_quat = self.container_states[:, 3:7]
            self.container_ang_vel = self.container_states[:, 10:13]
            
            # Compute container relative states in robot frame
            self.container_relative_position = quat_rotate_inverse(base_rot, self.container_pos - base_pos)
            self.container_relative_velocity = quat_rotate_inverse(base_rot, self.container_vel - base_vel)

    def _prepare_reward_function(self):
        """ Add load-specific rewards to the basic robot rewards """
        super()._prepare_reward_function()
        
        # Add load-specific reward functions
        if self.reward_scales.get("load_stability", 0.0) > 0.0:
            self.reward_functions.append(self._reward_load_stability)
            if "load_stability" not in self.reward_names:
                self.reward_names.append("load_stability")
        
        if self.reward_scales.get("load_position", 0.0) > 0.0:
            self.reward_functions.append(self._reward_load_position)
            if "load_position" not in self.reward_names:
                self.reward_names.append("load_position")
        
        # Add container-specific reward functions if enabled
        if self.has_container:
            if self.reward_scales.get("container_stability", 0.0) > 0.0:
                self.reward_functions.append(self._reward_container_stability)
                if "container_stability" not in self.reward_names:
                    self.reward_names.append("container_stability")
            
            if self.reward_scales.get("container_position", 0.0) > 0.0:
                self.reward_functions.append(self._reward_container_position)
                if "container_position" not in self.reward_names:
                    self.reward_names.append("container_position")

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            for ALL actors (robot, load, container).
        Args:
            env_ids (List[int]): Environment ids
        """
        if len(env_ids) == 0:
            return
             
        # Indices for accessing the main tensor
        env_ids_long = env_ids.long()
        
        # Reset robot state (using parent logic)
        if self.custom_origins:
            self.root_states[env_ids_long, 0, :] = self.base_init_state
            self.root_states[env_ids_long, 0, :3] += self.env_origins[env_ids_long, 0, :3]
            self.root_states[env_ids_long, 0, :2] += torch_rand_float(-1., 1., (len(env_ids_long), 2), device=self.device)
        else:
            self.root_states[env_ids_long, 0, :] = self.base_init_state
            self.root_states[env_ids_long, 0, :3] += self.env_origins[env_ids_long, 0, :3]
        
        # Robot velocities
        self.root_states[env_ids_long, 0, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids_long), 6), device=self.device)
        
        # Reset load state
        base_pos = self.root_states[env_ids_long, 0, 0:3]
        base_rot = self.root_states[env_ids_long, 0, 3:7]
        relative_load_pos = to_torch(self.cfg.load.init_pos, device=self.device).unsqueeze(0).repeat(len(env_ids_long), 1)
        world_load_pos = base_pos + quat_apply(base_rot, relative_load_pos)
        self.root_states[env_ids_long, 1, 0:3] = world_load_pos
        self.root_states[env_ids_long, 1, 3:7] = base_rot
        self.root_states[env_ids_long, 1, 7:13] = 0
        
        # Reset container state if enabled
        if self.has_container:
            relative_container_pos = to_torch(self.cfg.load.container_pos, device=self.device).unsqueeze(0).repeat(len(env_ids_long), 1)
            world_container_pos = base_pos + quat_apply(base_rot, relative_container_pos)
            self.root_states[env_ids_long, 2, 0:3] = world_container_pos
            self.root_states[env_ids_long, 2, 3:7] = torch.tensor([0., 0., 0., 1.], dtype=torch.float, device=self.device).repeat(len(env_ids_long), 1)
            self.root_states[env_ids_long, 2, 7:13] = 0

        # Update simulation state for all actors
        indices = torch.zeros(len(env_ids_long) * self.num_actors_per_env, dtype=torch.int32, device=self.device)
        for i in range(self.num_actors_per_env):
            indices[i*len(env_ids_long):(i+1)*len(env_ids_long)] = env_ids_long * self.num_actors_per_env + i
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(indices),
                                                    len(indices))

    def _reward_load_stability(self):
        """ Penalize angular velocity of the load """
        load_ang_vel = self.load_states[:, 10:13]
        ang_vel_penalty = torch.sum(load_ang_vel**2, dim=1)
        return torch.exp(-ang_vel_penalty / self.cfg.rewards.load_stability_sigma)

    def _reward_load_position(self):
        """ Penalize deviation from desired relative position """
        target_relative_pos = to_torch([0.0, 0.0, self.cfg.load.init_pos[2]], device=self.device)
        pos_error = torch.sum((self.load_relative_position - target_relative_pos)**2, dim=1)
        return torch.exp(-pos_error / self.cfg.rewards.load_position_sigma)

    def _reward_container_stability(self):
        """ Penalize container angular velocity """
        if self.container_states is not None:
            container_ang_vel = self.container_states[:, 10:13]
            ang_vel_penalty = torch.sum(container_ang_vel**2, dim=1)
            return torch.exp(-ang_vel_penalty / self.cfg.rewards.container_stability_sigma)
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_container_position(self):
        """ Penalize container position deviation """
        if self.container_states is not None:
            target_relative_pos = to_torch([0.0, 0.0, self.cfg.load.container_pos[2]], device=self.device)
            pos_error = torch.sum((self.container_relative_position - target_relative_pos)**2, dim=1)
            return torch.exp(-pos_error / self.cfg.rewards.container_position_sigma)
        return torch.zeros(self.num_envs, device=self.device)

    # Placeholder for required methods inherited but maybe needing adjustment
    def _get_env_origins(self):
        super()._get_env_origins()
        # Adjust env_origins shape if parent assumes single actor
        if self.env_origins.dim() == 2: # Shape (num_envs, 3)
             self.env_origins = self.env_origins.unsqueeze(1).repeat(1, self.num_actors_per_env, 1)
        elif self.env_origins.shape[1] != self.num_actors_per_env:
             print(f"Warning: env_origins shape {self.env_origins.shape} might be incompatible with num_actors {self.num_actors_per_env}")
             # Attempt to fix - assuming parent set origin for robot only
             if self.env_origins.shape[1] == 1:
                 self.env_origins = self.env_origins.repeat(1, self.num_actors_per_env, 1)
             
    def _get_noise_scale_vec(self):
        # Use parent implementation for now, might need adjustment
        # The parent calculates noise vec based on parent's obs space.
        # We concatenate load obs later, so this noise vec won't cover load obs.
        parent_noise_vec, parent_priv_noise_vec = super()._get_noise_scale_vec()
        
        # TODO: Extend noise vectors if load obs should have noise
        
        return parent_noise_vec, parent_priv_noise_vec

    def _init_height_points(self):
         return super()._init_height_points() # Use parent method
         
    def _process_rigid_shape_props(self, props, env_id):
         return super()._process_rigid_shape_props(props, env_id) # Use parent method

    def _process_dof_props(self, props, env_id):
         return super()._process_dof_props(props, env_id) # Use parent method
         
    def _process_rigid_body_props(self, props, env_id):
         return super()._process_rigid_body_props(props, env_id) # Use parent method

    def set_camera(self, position, lookat):
         super().set_camera(position, lookat) # Use parent method
         
    def create_sim(self):
         super().create_sim() # Use parent method, relies on our overridden _create_envs

    # --- Need to override reset logic ---
    def reset_idx(self, env_ids):
        # Reset DOFs first (uses parent method)
        self._reset_dofs(env_ids) 
        
        # Reset root states (uses OUR overridden method)
        self._reset_root_states(env_ids)

        # Resample commands (uses parent method)
        self._resample_commands(env_ids)

        # Reset buffers (mostly from parent, check logic)
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # Reset load/container specific things if needed
        # self.load_relative_position[env_ids] = 0 
        # etc.

        # Log episode info (uses parent method)
        self._log_episode_info(env_ids)

    def _log_episode_info(self, env_ids):
        """ Logs episode reward sums and custom curriculum info. """
        if len(env_ids) == 0: return
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # Filter out NaN rewards before mean
            valid_rewards = self.episode_sums[key][env_ids]
            mean_reward = torch.mean(valid_rewards[~torch.isnan(valid_rewards)])
            if torch.isnan(mean_reward): mean_reward = 0.0 # Handle case where all rewards are NaN
            self.extras["episode"]['rew_' + key] = mean_reward / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        # Log additional curriculum info if needed (from parent)
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels[env_ids].float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # Send timeout info
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[env_ids]

    def _reset_dofs(self, env_ids):
         super()._reset_dofs(env_ids) # Use parent method for robot DOFs

    def _resample_commands(self, env_ids):
         super()._resample_commands(env_ids) # Use parent method

    # Override _reset_root_states based on LeggedRobotBallBalance
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            for ALL actors (robot, load, container).
        Args:
            env_ids (List[int]): Environment ids
        """
        if len(env_ids) == 0:
             return
             
        # Indices for accessing the main tensor
        env_ids_long = env_ids.long() # Ensure long type for indexing
        
        # Robot state reset
        if self.custom_origins:
            self.root_states[env_ids_long, 0, :] = self.base_init_state # Use full state
            self.root_states[env_ids_long, 0, :3] += self.env_origins[env_ids_long, 0, :3] # Use robot origin
            self.root_states[env_ids_long, 0, :2] += torch_rand_float(-1., 1., (len(env_ids_long), 2), device=self.device)
        else:
            self.root_states[env_ids_long, 0, :] = self.base_init_state
            self.root_states[env_ids_long, 0, :3] += self.env_origins[env_ids_long, 0, :3]
        # Robot velocities
        self.root_states[env_ids_long, 0, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids_long), 6), device=self.device)
        
        # --- Load state reset ---
        # Get reset base position and orientation
        base_pos = self.root_states[env_ids_long, 0, 0:3]
        base_rot = self.root_states[env_ids_long, 0, 3:7]
        # Calculate desired world position for load (relative to robot base)
        relative_load_pos = to_torch(self.cfg.load.init_pos, device=self.device).unsqueeze(0).repeat(len(env_ids_long), 1)
        world_load_pos = base_pos + quat_apply(base_rot, relative_load_pos)
        # Set load state
        self.root_states[env_ids_long, 1, 0:3] = world_load_pos
        self.root_states[env_ids_long, 1, 3:7] = base_rot # Align with base rotation initially
        self.root_states[env_ids_long, 1, 7:13] = 0 # Zero velocities
        
        # --- Container state reset ---
        if self.has_container:
            relative_container_pos = to_torch(self.cfg.load.container_pos, device=self.device).unsqueeze(0).repeat(len(env_ids_long), 1)
            world_container_pos = base_pos + quat_apply(base_rot, relative_container_pos)
            # Set container state (static)
            self.root_states[env_ids_long, 2, 0:3] = world_container_pos
            self.root_states[env_ids_long, 2, 3:7] = torch.tensor([0., 0., 0., 1.], dtype=torch.float, device=self.device).repeat(len(env_ids_long), 1) # Identity quat
            self.root_states[env_ids_long, 2, 7:13] = 0 # Zero velocities

        # Set the state in the simulation tensor using the full tensor
        # Calculate indices for ALL actors in the selected envs
        indices = torch.zeros(len(env_ids_long) * self.num_actors_per_env, dtype=torch.int32, device=self.device)
        for i in range(self.num_actors_per_env):
             indices[i*len(env_ids_long):(i+1)*len(env_ids_long)] = env_ids_long * self.num_actors_per_env + i
             
        # Ensure the full state tensor is contiguous before updating sim
        self.root_states = self.root_states.contiguous()
        
        # Update simulation state
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states), # Use the full updated tensor
                                                     gymtorch.unwrap_tensor(indices),
                                                     len(indices))
                                                     
    # --- Methods that likely need review ---
    # post_physics_step: Needs to use self.root_states correctly
    # compute_observations: Needs to use self.root_states correctly
    # _prepare_reward_function: Should be fine if reward functions are correct
    # Individual reward functions: Need to use correct states
    
    # Copied/Adapted from previous versions
    def post_physics_step(self):
        """ Processes the physics step to compute observations, rewards, resets, and applies external forces
        """
        # Refresh all tensors first
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Update common quantities (needed before reward/obs computation)
        # These use self.root_states with correct actor indices
        self.base_quat[:] = self.root_states[:, 0, 3:7]  # Robot is actor 0
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Call parent's callback for things like feet state computation etc.
        self._post_physics_step_callback()
        
        # Update load state (actor 1)
        self.load_states = self.root_states[:, 1].clone()  # Get full load state
        self.load_pos = self.load_states[:, :3]
        self.load_vel = self.load_states[:, 7:10]
        self.load_quat = self.load_states[:, 3:7]
        self.load_ang_vel = self.load_states[:, 10:13]

        # Update relative positions and velocities in robot frame
        base_pos = self.root_states[:, 0, :3]
        base_rot = self.root_states[:, 0, 3:7]
        base_vel = self.root_states[:, 0, 7:10]

        # Compute load relative states in robot frame
        self.load_relative_position = quat_rotate_inverse(base_rot, self.load_pos - base_pos)
        self.load_relative_velocity = quat_rotate_inverse(base_rot, self.load_vel - base_vel)

        # Update container state if enabled (actor 2)
        if self.has_container:
            self.container_states = self.root_states[:, 2].clone()  # Get full container state
            self.container_pos = self.container_states[:, :3]
            self.container_vel = self.container_states[:, 7:10]
            self.container_quat = self.container_states[:, 3:7]
            self.container_ang_vel = self.container_states[:, 10:13]
            
            # Compute container relative states in robot frame
            self.container_relative_position = quat_rotate_inverse(base_rot, self.container_pos - base_pos)
            self.container_relative_velocity = quat_rotate_inverse(base_rot, self.container_vel - base_vel)
            
        # Compute terminations, rewards, observations
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # Compute obs AFTER possible reset

        # Update last actions/velocities for next step
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]  # Robot only
        
    def compute_observations(self):
        """ Computes observations by calling parent and adding load info.
        """
        # Call parent's compute_observations to get robot obs
        # This sets self.obs_buf based on robot state
        super().compute_observations()
        
        # Add load observations
        load_obs = torch.cat((
            self.load_relative_position * self.obs_scales.lin_vel,
            self.load_relative_velocity * self.obs_scales.lin_vel
        ), dim=-1)
        self.obs_buf = torch.cat((self.obs_buf, load_obs), dim=-1)
        
        # Add container observations if enabled
        if self.has_container and self.container_relative_position is not None:
            container_obs = torch.cat((
                self.container_relative_position * self.obs_scales.lin_vel,
                self.container_relative_velocity * self.obs_scales.lin_vel
            ), dim=-1)
            self.obs_buf = torch.cat((self.obs_buf, container_obs), dim=-1)
        
        # Parent's compute_observations handles noise addition if self.add_noise is True
        return self.obs_buf

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, adding load/container rewards. """
        # Call parent implementation first
        super()._prepare_reward_function()
        
        # Add load-specific reward functions
        if self.reward_scales.get("load_stability", 0.0) > 0.0:
            self.reward_functions.append(self._reward_load_stability)
            # Ensure name matches if not already added by parent (it shouldn't be)
            if "load_stability" not in self.reward_names:
                 self.reward_names.append("load_stability")
        
        if self.reward_scales.get("load_position", 0.0) > 0.0:
            self.reward_functions.append(self._reward_load_position)
            if "load_position" not in self.reward_names:
                 self.reward_names.append("load_position")
        
        # Add container-specific reward functions if enabled
        if self.has_container:
            if self.reward_scales.get("container_stability", 0.0) > 0.0:
                self.reward_functions.append(self._reward_container_stability)
                if "container_stability" not in self.reward_names:
                     self.reward_names.append("container_stability")
            
            if self.reward_scales.get("container_position", 0.0) > 0.0:
                self.reward_functions.append(self._reward_container_position)
                if "container_position" not in self.reward_names:
                     self.reward_names.append("container_position")

    # --- Define the actual reward functions ---
    def _reward_load_stability(self):
        # Penalizes angular velocity of the load
        load_ang_vel = self.load_states[:, 10:13]
        ang_vel_penalty = torch.sum(load_ang_vel**2, dim=1)
        # Use sigma from config for scaling
        return torch.exp(-ang_vel_penalty / self.cfg.rewards.load_stability_sigma)

    def _reward_load_position(self):
        # Penalizes deviation from desired relative position (e.g., centered above base)
        # Target is likely [0, 0, height_offset]
        target_relative_pos = to_torch([0.0, 0.0, self.cfg.load.init_pos[2]], device=self.device)
        pos_error = torch.sum((self.load_relative_position - target_relative_pos)**2, dim=1)
        # Use sigma from config for scaling
        return torch.exp(-pos_error / self.cfg.rewards.load_position_sigma)

    def _reward_container_stability(self):
        # Container is static, angular velocity should be zero
        if self.container_states is not None:
             container_ang_vel = self.container_states[:, 10:13]
             ang_vel_penalty = torch.sum(container_ang_vel**2, dim=1)
             # Use sigma from config for scaling
             return torch.exp(-ang_vel_penalty / self.cfg.rewards.container_stability_sigma)
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_container_position(self):
        # Penalizes deviation from desired relative position
        if self.container_states is not None:
             target_relative_pos = to_torch([0.0, 0.0, self.cfg.load.container_pos[2]], device=self.device)
             pos_error = torch.sum((self.container_relative_position - target_relative_pos)**2, dim=1)
             # Use sigma from config for scaling
             return torch.exp(-pos_error / self.cfg.rewards.container_position_sigma)
        return torch.zeros(self.num_envs, device=self.device)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions with shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Torques sent to the simulation with shape (num_envs, num_dof)
        """
        # Ensure actions have the right shape
        actions_scaled = actions * self.cfg.control.action_scale
        
        # Handle the case where num_actions != num_dof
        if actions_scaled.shape[1] != self.num_dof:
            # Assuming first num_dof actions correspond to the DOFs
            actions_scaled = actions_scaled[:, :self.num_dof]
        
        # pd controller
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
            
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

# --- Remove the erroneous linking lines --- 