import os
import sys
from typing import Dict

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.terrain import Terrain

import numpy as np

import random

class PointFootWithLoadBalance:
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg()
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_propriceptive_obs
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.num_actors = cfg.env.num_actors

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.proprioceptive_obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        # self.num_balls = self.cfg.env.num_balls
        self.multi_balls = False


        self.num_balls = cfg.env.num_actors - 1
        self.num_obs_load = cfg.env.num_obs_load

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None


        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self._include_feet_height_rewards = self._check_if_include_feet_height_rewards()
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True




    def get_load_observations(self):
        ''' Return the load observations
        '''
        return self.load_observations

    def get_observations_history(self):
        ''' Overide the base class method to return the observations and observation history
        '''
        return self.obs_history_buf

    def reset(self):
        ''' Reset all robots (overide the base class method)
        '''
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs,



    def get_observations(self):
        return self.proprioceptive_obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.proprioceptive_obs_buf = torch.clip(self.proprioceptive_obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.proprioceptive_obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.obs_history_buf, self.load_observations

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:,0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_positions = self.root_states[:, 0, 0:3]
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.measured_heights = self._get_heights()

        # prepare quantities
        if self.multi_balls:
            self.ball_positions = self.root_states[:, 1:, 0:3]
            self.ball_orientations = self.root_states[:, 1:, 3:7]
            self.ball_linvels = self.root_states[:, 1:, 7:10]
            self.ball_angvels = self.root_states[:, 1:, 10:13]
        else:
            self.ball_positions = self.root_states[:, 1, 0:3]
            self.ball_orientations = self.root_states[:, 1, 3:7]
            self.ball_linvels = self.root_states[:, 1, 7:10]
            self.ball_angvels = self.root_states[:, 1, 10:13]

        self._compute_feet_states()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.update_load_state_buffer()
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]

        self.last_wrenches_on_robot[:] = self.wrenches_on_robot

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _check_if_include_feet_height_rewards(self):
        members = [attr for attr in dir(self.cfg.rewards.scales) if not attr.startswith("__")]
        for scale in members:
            if "feet_height" in scale:
                return True
        return False

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # tot_xy_too_large = False
        if self.multi_balls:
            for i in range(self.num_balls):
                ball_x = self.root_states[:, i+1, 0]
                ball_y = self.root_states[:, i+1, 1]
                base_x = self.root_states[:, 0, 0]
                base_y = self.root_states[:, 0, 1]
                x_difference = base_x - ball_x
                y_difference = base_y - ball_y
                # x_too_large 是 x_difference > 0.3 或者 <-0.3
                # y_too_large 是 y_difference > 0.4 或者 <-0.4
                xy_too_large = ((x_difference > 0.3) | (
                    x_difference < -0.3)) | ((y_difference > 0.4) | (y_difference < -0.4))
                
                # tot_xy_too_large |= xy_too_large
                
                # if torch.any(xy_too_large):
                #     print("break")
                #     break
            
                # print(f"xy_too_large!!!!!: {xy_too_large}")

                self.reset_buf |= xy_too_large
        else:
            ball_x = self.root_states[:, 1, 0]
            ball_y = self.root_states[:, 1, 1]
            base_x = self.root_states[:, 0, 0]
            base_y = self.root_states[:, 0, 1]
            x_difference = base_x - ball_x
            y_difference = base_y - ball_y
            xy_too_large = ((x_difference > 0.3) | (
                x_difference < -0.3)) | ((y_difference > 0.4) | (y_difference < -0.4))
            self.reset_buf |= xy_too_large

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample(env_ids)

        self.actions[env_ids] = 0. # prevent actions at the first frame make the leg and contact plate collide
        
        self._reset_buffers(env_ids)
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.obs_history_buf[env_ids, :] = 0. # reset obs history
        self.load_observations[env_ids, :] = 0.
        self.load_state_buffer[env_ids] = torch.zeros(
            self.load_state_buffer[env_ids].shape, device=self.device)
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_air_time[env_ids] = 0.
        self.current_max_feet_height[env_ids] = 0.
        self.last_max_feet_height[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


    def update_load_state_buffer(self):
        ''' Update the load states buffer
        '''
        current_load_state = torch.cat((quat_rotate_inverse(self.base_quat, self.ball_positions - self.base_positions), # 小球相对于机器人的位置
                                  quat_rotate_inverse(self.base_quat, self.ball_linvels - self.root_states[:, 0, 7:10]), # 小球的相对机器人的线速度
                                  self.load_mass.unsqueeze(1),
                                  self.load_friction.unsqueeze(1),
                                    ), dim=-1)
        
        # 更新缓冲区，保存过去20帧的wrench, 0-20 由旧到新, 最末端是最新的
        self.load_state_buffer = torch.roll(self.load_state_buffer, shifts=-1, dims=1)
        self.load_state_buffer[:, -1, :] = current_load_state

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # def compute_observations(self):
    #     """ Computes observations
    #     """
    #     self.compute_proprioceptive_observations()
    #     self.compute_privileged_observations()

    #     self._add_noise_to_obs()

    def compute_observations(self):
        """ Computes observations
        """
        self.proprioceptive_obs_buf = self.compute_proprioceptive_observations()
        self.obs_history_buf = torch.cat((self.obs_history_buf[:, self.proprioceptive_obs_buf.shape[1]:], 
                                          self.proprioceptive_obs_buf), dim=1) 
        # add noise to proprioceptive observations
        if self.add_noise:

            self.proprioceptive_obs_buf += (2 * torch.rand_like(self.proprioceptive_obs_buf) - 1) * self.noise_scale_vec[0]

        if self.cfg.env.num_privileged_obs is not None:
            print(f"self.p_gains shape: {self.p_gains.shape}")
            print(f"self.d_gains shape: {self.d_gains.shape}")
            print(f"self.friction_coeffs_tensor shape: {self.friction_coeffs_tensor.shape}")
            print(f"self.leg_params_tensor shape: {self.leg_params_tensor.shape}")
            print(f"self.mass_params_tensor shape: {self.mass_params_tensor.shape}")
            print(f"self.motor_strength shape: {self.motor_strength.shape}")
            self.adapt_observations = torch.cat((
                                        self.p_gains.expand(self.num_envs, -1),#12
                                        self.d_gains.expand(self.num_envs, -1),#12
                                        self.friction_coeffs_tensor,#1
                                        self.leg_params_tensor,#4
                                        self.mass_params_tensor,#10
                                        self.motor_strength[0] - 1, #12
                                        self.motor_strength[1] - 1 #12
                                        ),dim=-1)
            
            self.load_observations = torch.cat((
                                    quat_rotate_inverse(self.base_quat, 
                                                        self.ball_positions - self.base_positions), # 小球相对于机器人的位置
                                    quat_rotate_inverse(self.base_quat, 
                                                        self.ball_linvels - self.root_states[:, 0, 7:10]), # 小球的相对机器人的线速度
                                    self.load_mass.unsqueeze(1),
                                    self.load_friction.unsqueeze(1),
                                    ), dim=-1)
            valid_indices = self.load_observations[:,6] > 0.1
            self.load_observations[~valid_indices] = 0.
            
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:,0, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements

            self.privileged_obs_buf = torch.cat((
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.proprioceptive_obs_buf,
                    heights,
                    self.adapt_observations,
                    self.torques,
                    (self.last_dof_vel - self.dof_vel) / self.dt,
                    self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                    ),dim=-1)
            
    def compute_proprioceptive_observations(self):
        """ Computes privileged observations
        """
        self.proprioceptive_obs_buf = torch.cat((  
                                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands[:, :3] * self.commands_scale,
                                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                self.actions
                                                ),dim=-1)
        return self.proprioceptive_obs_buf
    
    def _add_noise_to_obs(self):
        # add noise if needed
        if self.add_noise:
            obs_noise_vec, privileged_extra_obs_noise_vec = self.noise_scale_vec
            obs_noise_buf = (2 * torch.rand_like(self.proprioceptive_obs_buf) - 1) * obs_noise_vec
            self.proprioceptive_obs_buf += obs_noise_buf
            if self.num_privileged_obs is not None:
                privileged_extra_obs_buf = (2 * torch.rand_like(
                    self.privileged_obs_buf[:, len(self.noise_scale_vec[0]):]) - 1) * privileged_extra_obs_noise_vec
                self.privileged_obs_buf += torch.cat((obs_noise_buf, privileged_extra_obs_buf), dim=1)

    def compute_privileged_observations(self):
        if self.num_privileged_obs is not None:
            self._compose_privileged_obs_buf_no_height_measure()
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights_critic:
                self.privileged_obs_buf = self._add_height_measure_to_buf(self.privileged_obs_buf)
            if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
                raise RuntimeError(
                    f"privileged_obs_buf size ({self.privileged_obs_buf.shape[1]}) does not match num_privileged_obs ({self.num_privileged_obs})")

    def _compose_privileged_obs_buf_no_height_measure(self):
        self.privileged_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                             self.projected_gravity,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions,
                                             self.commands[:, :3] * self.commands_scale,
                                             ), dim=-1)

    # def compute_proprioceptive_observations(self):
    #     self._compose_proprioceptive_obs_buf_no_height_measure()
    #     if self.cfg.terrain.measure_heights_actor:
    #         self.proprioceptive_obs_buf = self._add_height_measure_to_buf(self.proprioceptive_obs_buf)
    #     if self.proprioceptive_obs_buf.shape[1] != self.num_obs:
    #         raise RuntimeError(
    #             f"obs_buf size ({self.proprioceptive_obs_buf.shape[1]}) does not match num_obs ({self.num_obs})")

    def _add_height_measure_to_buf(self, buf):
        heights = torch.clip(self.root_states[:, 0, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                             1.) * self.obs_scales.height_measurements
        buf = torch.cat(
            (buf, heights), dim=-1
        )
        return buf

    def _compose_proprioceptive_obs_buf_no_height_measure(self):
        self.proprioceptive_obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                                 self.projected_gravity,
                                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                 self.dof_vel * self.obs_scales.dof_vel,
                                                 self.actions,
                                                 self.commands[:, :3] * self.commands_scale,
                                                 ), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_load_rigid_shape_props(self, props, env_id):
        """ Randomize the friction coefficient of the load 
        """
        friction_coe_range = self.cfg.load_params.friction_coefficient_range
        props[0].friction = np.random.uniform(friction_coe_range[0], friction_coe_range[1])
        return props
    
    def _process_load_rigid_body_props(self, props, env_id):
        """ Randomize the load mass
        """
        rng = self.cfg.load_params.mass_range
        props[0].mass = np.random.uniform(rng[0], rng[1])   
        return props
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    # def _process_rigid_body_props(self, props, env_id):
    #     """ Process rigid body properties for domain randomization 
        
    #     Modified to handle mass randomization and center of mass (COM) randomization:
    #     1. Mass Randomization:
    #        - Checks if randomize_base_mass is enabled in config
    #        - Uses added_mass_range from config to add random mass
    #        - Validates mass range has exactly 2 elements [min, max]
    #        - Prints warning if range is invalid
        
    #     2. COM Randomization:
    #        - Checks if randomize_base_com is enabled in config
    #        - Uses rand_com_vec from config for x,y,z randomization
    #        - Adds random offset to COM within specified range
           
    #     Args:
    #         props (List[gymapi.RigidBodyProperties]): Properties of each rigid body
    #         env_id (int): Environment id
            
    #     Returns:
    #         List[gymapi.RigidBodyProperties]: Modified rigid body properties
    #     """
    #     if self.cfg.domain_rand.randomize_base_mass:
    #         rng = self.cfg.domain_rand.added_mass_range
    #         if len(rng) == 2:  # Check if range is properly defined
    #             props[0].mass += np.random.uniform(rng[0], rng[1])
    #         else:
    #             print("Warning: Invalid mass randomization range, skipping mass randomization")
    #     if self.cfg.domain_rand.randomize_base_com:
    #         com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
    #         props[0].com.x += np.random.uniform(-com_x, com_x)
    #         props[0].com.y += np.random.uniform(-com_y, com_y)
    #         props[0].com.z += np.random.uniform(-com_z, com_z)
    #     return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        # randomize base mass
        if hasattr(self.cfg.domain_rand, 'randomize_base_mass') and self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1], size=(1, ))
            props[0].mass += np.random.uniform(rng[0], rng[1])
        else:
            rand_mass = np.zeros((1, ))
        # randomize base com
        if hasattr(self.cfg.domain_rand, 'randomize_base_com') and self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_mass_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros((3))
        # randomize base inertia
        if hasattr(self.cfg.domain_rand, 'randomize_base_inertia') and self.cfg.domain_rand.randomize_base_inertia:
            rng_inertia_xx = self.cfg.domain_rand.added_inertia_range_xx
            rng_inertia_xy = self.cfg.domain_rand.added_inertia_range_xy
            rng_inertia_xz = self.cfg.domain_rand.added_inertia_range_xz
            rng_inertia_yy = self.cfg.domain_rand.added_inertia_range_yy
            rng_inertia_zz = self.cfg.domain_rand.added_inertia_range_zz
            rand_xx = np.random.uniform(rng_inertia_xx[0], rng_inertia_xx[1], size=(1, ))
            rand_xy = np.random.uniform(rng_inertia_xy[0], rng_inertia_xy[1], size=(1, ))
            rand_xz = np.random.uniform(rng_inertia_xz[0], rng_inertia_xz[1], size=(1, ))
            rand_yy = np.random.uniform(rng_inertia_yy[0], rng_inertia_yy[1], size=(1, ))
            rand_zz = np.random.uniform(rng_inertia_zz[0], rng_inertia_zz[1], size=(1, ))
            rand_inertia = np.concatenate([rand_xx,rand_xy,rand_xz,rand_yy,np.array([0]),rand_zz])
            rand_inertia_matrix = gymapi.Mat33() #Mat33 style
            rand_inertia_matrix.x = gymapi.Vec3(rand_inertia[0],rand_inertia[1],rand_inertia[2])
            rand_inertia_matrix.y = gymapi.Vec3(rand_inertia[1],rand_inertia[3],rand_inertia[4])
            rand_inertia_matrix.z = gymapi.Vec3(rand_inertia[2],rand_inertia[4],rand_inertia[5])
            props[0].inertia.x += rand_inertia_matrix.x
            props[0].inertia.y += rand_inertia_matrix.y
            props[0].inertia.z += rand_inertia_matrix.z
        else:
            rand_inertia = np.zeros((6))

        # randomize leg mass
        if hasattr(self.cfg.domain_rand, 'randomize_leg_mass') and self.cfg.domain_rand.randomize_leg_mass:
            rng_leg = self.cfg.domain_rand.added_leg_mass_range
            factor_leg_mass = self.cfg.domain_rand.factor_leg_mass_range
            rand_leg_mass = np.random.uniform(factor_leg_mass[0], factor_leg_mass[1], size=(1, ))
            for i in range(1,19):
                props[i].mass *= np.random.uniform(factor_leg_mass[0], factor_leg_mass[1])
        else:
            rand_leg_mass = np.zeros((1, ))
  
        # randomize leg com
        if hasattr(self.cfg.domain_rand, 'randomize_leg_com') and self.cfg.domain_rand.randomize_leg_com:
            rng_leg_com = self.cfg.domain_rand.added_leg_com_range
            rand_leg_com = np.random.uniform(rng_leg_com[0], rng_leg_com[1], size=(3, ))
            props[1].com += gymapi.Vec3(*rand_leg_com)
            props[2].com += gymapi.Vec3(*rand_leg_com)
            props[3].com += gymapi.Vec3(*rand_leg_com)

            props[5].com += gymapi.Vec3(*rand_leg_com)
            props[6].com += gymapi.Vec3(*rand_leg_com)
            props[7].com += gymapi.Vec3(*rand_leg_com)

            props[11].com += gymapi.Vec3(*rand_leg_com)
            props[12].com += gymapi.Vec3(*rand_leg_com)
            props[13].com += gymapi.Vec3(*rand_leg_com)

            props[15].com += gymapi.Vec3(*rand_leg_com)
            props[16].com += gymapi.Vec3(*rand_leg_com)
            props[17].com += gymapi.Vec3(*rand_leg_com)
        else:
            rand_leg_com = np.zeros((3))

        mass_params = np.concatenate([rand_mass, rand_com, rand_inertia])
        leg_params =  np.concatenate([rand_leg_mass, rand_leg_com])
        return props, mass_params, leg_params

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample(self, env_ids):
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
                                                         self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * (
                    actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        # env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.all_robot_indices = self.num_actors * \
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        robot_indices = self.all_robot_indices[env_ids].flatten()
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(robot_indices), len(robot_indices))


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids, 0, :] = self.base_init_state[env_ids, :]
            self.root_states[env_ids, :, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state[env_ids]
            self.root_states[env_ids, :, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 0, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        if self.multi_balls:
            for i in range(1, self.num_balls+1):
                self.root_states[env_ids, i,:] = self.ball_init_state[env_ids]  # ball init
                self.root_states[env_ids, 
                                    i,
                                    0:2] = self.root_states[env_ids,
                                                            0,
                                                            0:2] + torch_rand_float(-0.15,
                                                                                    0.15,
                                                                                    (len(env_ids),
                                                                                    2),
                                                                                    device=self.device) # 定义小球初始 x,y 位置
                self.root_states[env_ids,
                                    i,
                                    2] = self.root_states[env_ids,
                                                        0,
                                                        2] + torch.rand(len(env_ids),
                                                                        device=self.device) * 0.01 + self.load_radius + 0.2 # 定义小球初始 z
                self.root_states[env_ids,
                                i,
                                7:13] = torch_rand_float(-0.5,
                                                        0.5,
                                                        (len(env_ids),
                                                        6),
                                                        device=self.device) # 定义小球初始速度
        else:
            self.root_states[env_ids, 1,:] = self.ball_init_state[env_ids]  # ball init
            self.root_states[env_ids, 
                                1,
                                0:2] = self.root_states[env_ids,
                                                        0,
                                                        0:2] + torch_rand_float(-0.15,
                                                                                0.15,
                                                                                (len(env_ids),
                                                                                2),
                                                                                device=self.device) # 定义小球初始 x,y 位置
            self.root_states[env_ids,
                                1,
                                2] = self.root_states[env_ids,
                                                    0,
                                                    2] + torch.rand(len(env_ids),
                                                                    device=self.device) * 0.01 + self.load_radius + 0.2 # 定义小球初始 z
            self.root_states[env_ids,
                            1,
                            7:13] = torch_rand_float(-0.5,
                                                    0.5,
                                                    (len(env_ids),
                                                    6),
                                                    device=self.device) # 定义小球初始速度

        
        
        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.all_actor_indices = torch.arange(
            self.num_actors * self.num_envs,
            dtype=torch.int32,
            device=self.device).view(
            self.num_envs,
            self.num_actors)
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.root_states),
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
    

    def _push_robots(self):
        """Random pushes the robots."""
        max_push_force = (
                self.base_mass.mean().item()
                * self.cfg.domain_rand.max_push_vel_xy
                / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force, max_push_force, (self.num_envs, 3), device=self.device
        )
        self.rigid_body_external_forces[:, 0, 0:3] = quat_rotate(
            self.base_quat, rigid_body_external_forces
        )
        self.rigid_body_external_forces[:, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, 0, :2] - self.env_origins[env_ids, 0, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids, 0, :] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        obs_noise_vec = torch.zeros(self.cfg.env.num_propriceptive_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        obs_noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        obs_noise_vec[3:6] = noise_scales.gravity * noise_level
        command_end_idx = 6 + self.cfg.commands.num_commands
        obs_noise_vec[6:command_end_idx] = 0.  # commands
        dof_pos_end_idx = command_end_idx + self.num_dof
        obs_noise_vec[command_end_idx:dof_pos_end_idx] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        dof_vel_end_idx = dof_pos_end_idx + self.num_dof
        obs_noise_vec[dof_pos_end_idx:dof_vel_end_idx] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        last_action_end_idx = dof_vel_end_idx + self.num_actions
        obs_noise_vec[dof_vel_end_idx:last_action_end_idx] = 0.  # previous actions
        if self.cfg.env.num_privileged_obs is not None:
            privileged_extra_obs_noise_vec = torch.zeros(
                self.cfg.env.num_privileged_obs - self.cfg.env.num_propriceptive_obs, device=self.device)
        else:
            privileged_extra_obs_noise_vec = None

        if self.cfg.terrain.measure_heights_actor:
            measure_heights_end_idx = last_action_end_idx + len(self.cfg.terrain.measured_points_x) * len(
                self.cfg.terrain.measured_points_y)
            obs_noise_vec[
            last_action_end_idx:measure_heights_end_idx] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        if self.cfg.terrain.measure_heights_critic:
            if self.cfg.env.num_privileged_obs is not None:
                privileged_extra_obs_noise_vec[
                :len(self.cfg.terrain.measured_points_x) * len(
                    self.cfg.terrain.measured_points_y)] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return obs_noise_vec, privileged_extra_obs_noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(
            self.num_envs, self.num_actors, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 0, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis
        
        if hasattr(self.cfg.domain_rand, 'randomize_action_delay') and self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.motor_offsets = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.obs_history_length * self.proprioceptive_obs_buf.shape[-1], 
                                           dtype=torch.float, device=self.device)
        self.load_observations = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device)

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                              device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                        device=self.device, requires_grad=False)
        self.first_contact = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.last_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                device=self.device, requires_grad=False)
        self.current_max_feet_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                                   device=self.device, requires_grad=False)
        self.rigid_body_external_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )
        self.rigid_body_external_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False
        )

        # wrench related data
        self.wrenches_on_robot = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device) # 6-dimensional wrench on robot by load
        self.wrench_buffer = torch.zeros(
            self.num_envs, 50, 6, dtype=torch.float, device=self.device, requires_grad=False) # 20-dimensional wrench buffer
        self.last_wrenches_on_robot = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        
        # load state related data
        self.load_state_buffer = torch.zeros(
            self.num_envs, 20, 8, dtype=torch.float, device=self.device, requires_grad=False) # 

        # initialize ball related data
        if self.multi_balls:
            self.ball_positions = self.root_states[..., 1:, 0:3] # (num_envs, num_balls, 3)
            self.ball_orientations = self.root_states[..., 1:, 3:7] # (num_envs, num_balls, 3)
            self.ball_linvels = self.root_states[..., 1:, 7:10] # (num_envs, num_balls, 3)
            self.ball_angvels = self.root_states[..., 1:, 10:13] # (num_envs, num_balls, 3)
        else:    
            self.ball_positions = self.root_states[..., 1, 0:3] # (num_envs, num_balls, 3)
            self.ball_orientations = self.root_states[..., 1, 3:7] # (num_envs, num_balls, 3)
            self.ball_linvels = self.root_states[..., 1, 7:10] # (num_envs, num_balls, 3)
            self.ball_angvels = self.root_states[..., 1, 10:13] # (num_envs, num_balls, 3)


        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights_actor or self.cfg.terrain.measure_heights_critic:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]


        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        print("asset_path: ",asset_path)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) + self.num_balls
        print("self.num_bodies: ", self.num_bodies)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names) + self.num_balls
        print("self.num_bodies 2: ", self.num_bodies)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        # self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # # Do we need this?
        # self.base_mass = torch.zeros(
        #     self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        # )
        

        ball_init_state_list = self.cfg.ball_init_state.pos + self.cfg.ball_init_state.rot + \
            self.cfg.ball_init_state.lin_vel + self.cfg.ball_init_state.ang_vel
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + \
            self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        ball_init_state_list = self.cfg.ball_init_state.pos + self.cfg.ball_init_state.rot + \
            self.cfg.ball_init_state.lin_vel + self.cfg.ball_init_state.ang_vel
        # self.base_init_state = to_torch(self.num_envs,base_init_state_list, device=self.device, requires_grad=False)
        # self.ball_init_state = to_torch(ball_init_state_list, device=self.device, requires_grad=False)
        self.base_init_state = torch.zeros(
            self.num_envs, 13, device=self.device)
        
        for idx in range(self.num_envs):
            self.base_init_state[idx] = to_torch(
                base_init_state_list, device=self.device, requires_grad=False)

        self.ball_init_state = torch.zeros(
            self.num_envs, 13, device=self.device)
        
        for idx in range(self.num_envs):
            self.ball_init_state[idx] = to_torch(
                ball_init_state_list, device=self.device, requires_grad=False)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[0, :3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.load_handles = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device, requires_grad=False)
        self.leg_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Ball related data

        self.obj_handles = []
        self.load_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.load_friction = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        

        for i in range(self.num_envs):
            # create ball instance
            self.load_radius = 0.05 * random.uniform(0.5, 3.)
            ball_options = gymapi.AssetOptions()
            ball_options.density = 4 / (self.load_radius**3)
            ball_asset = self.gym.create_box(self.sim, self.load_radius, self.load_radius, 
                            self.load_radius, ball_options)
            
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i, 0, :].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # body_props = self._process_rigid_body_props(body_props, i)
            body_props, mass_params, leg_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
            self.leg_params_tensor[i, :] = torch.from_numpy(leg_params).to(self.device).to(torch.float)



            # Ball additions to the envs and set friction properties
            load_rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
                ball_asset)
            load_rigid_shape_props = self._process_load_rigid_shape_props(
                load_rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                ball_asset, load_rigid_shape_props)
            self.load_friction[i] = load_rigid_shape_props[0].friction
                   
            start_pose_ball = gymapi.Transform()
            start_pose_ball.p = gymapi.Vec3(
                pos[0], pos[1], pos[2] + self.load_radius + 0.01)
            
            # Create ball instance
            ball_handle = self.gym.create_actor(
                env_handle, ball_asset, start_pose_ball, 'ball', i, 0, 0)
            num_actors = self.gym.get_actor_count(env_handle)
            assert num_actors == self.cfg.env.num_actors, f"number of actors in the environment {i} is {num_actors}, expected {self.cfg.env.num_actors}"

            # Get and process rigid body (ball or cube) properties mass
            load_props = self.gym.get_actor_rigid_body_properties(
                env_handle, ball_handle)
            load_props = self._process_load_rigid_body_props(load_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, ball_handle, load_props, recomputeInertia=True)
            
            self.load_mass[i] = load_props[0].mass
            
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.load_handles.append(ball_handle)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs,
                self.num_actors, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                           (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(
                torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:, 0, :] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 
                self.num_actors, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 0, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 0, 2] = 0.

    def _parse_cfg(self):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights_actor and not self.terrain.cfg.measure_heights_critic:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, 0, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, 0, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, 0, :3]).unsqueeze(1)

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_heights_below_foot(self):
        """ Samples heights of the terrain at required points around each foot.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.feet_state[:, :, :2]

        heights = self._get_terrain_heights_from_points(points)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_terrain_heights_from_points(self, points):
        points = points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights

    def _compute_feet_states(self):
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.last_feet_air_time = self.feet_air_time * self.first_contact + self.last_feet_air_time * ~self.first_contact
        self.feet_air_time *= ~self.contact_filt
        if self._include_feet_height_rewards:
            self.last_max_feet_height = self.current_max_feet_height * self.first_contact + self.last_max_feet_height * ~self.first_contact
            self.current_max_feet_height *= ~self.contact_filt
            self.feet_height = self.feet_state[:, :, 2] - self._get_heights_below_foot()
            self.current_max_feet_height = torch.max(self.current_max_feet_height,
                                                     self.feet_height)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 0, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.),
            dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward steps between proper duration
        rew_airTime_below_min = torch.sum(
            torch.min(self.feet_air_time - self.cfg.rewards.min_feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime_above_max = torch.sum(
            torch.min(self.cfg.rewards.max_feet_air_time - self.feet_air_time,
                      torch.zeros_like(self.feet_air_time)) * self.first_contact,
            dim=1)
        rew_airTime = rew_airTime_below_min + rew_airTime_above_max
        return rew_airTime

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact

    def _reward_unbalance_feet_air_time(self):
        return torch.var(self.last_feet_air_time, dim=-1)

    def _reward_unbalance_feet_height(self):
        return torch.var(self.last_max_feet_height, dim=-1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                         5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize displacement and rotation at zero commands
        reward_lin = torch.abs(self.base_lin_vel[:, :2]) * (torch.abs(self.commands[:, :2] < 0.1))
        reward_ang = (torch.abs(self.base_ang_vel[:, -1]) * (torch.abs(self.commands[:, 2] < 0.1))).unsqueeze(dim=-1)
        return torch.sum(torch.cat((reward_lin, reward_ang), dim=-1), dim=-1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_distance(self):
        reward = 0
        for i in range(self.feet_state.shape[1] - 1):
            for j in range(i + 1, self.feet_state.shape[1]):
                feet_distance = torch.norm(
                    self.feet_state[:, i, :2] - self.feet_state[:, j, :2], dim=-1
                )
            reward += torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        return (~self.reset_buf).float() * self.dt

    # def _reward_symmetry_feet_force(self):
    #     # penalize laterally nonsymmetric feet contact forces
    #     # Body names: ['base',
    #     # 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
    #     # 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
    #     # 'Head_upper', 'Head_lower',
    #     # 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot',
    #     # 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
    #     # feet_indices: tensor([ 4,  8, 14, 18], device='cuda:0')

    #     fl_foot_force = self.contact_forces[:, 4, :]
    #     rl_foot_force = self.contact_forces[:, 14, :]
    #     fr_foot_force = self.contact_forces[:, 8, :]
    #     rr_foot_force = self.contact_forces[:, 18, :]
    #     return torch.sum(torch.square(fl_foot_force - fr_foot_force) + torch.square(rl_foot_force - rr_foot_force), dim=1)

    def _reward_ball_lin_vel(self):
        """ Penalize the ball's linear velocity """
        tot_ball_vel_rew = 0

        if self.multi_balls:
            for i in range(self.num_balls):
                # 相对机器人的速度
                ball_vel = torch.norm(quat_rotate_inverse(self.base_quat,(self.ball_linvels[:,i,:]- 
                                                                        self.root_states[:, 0, 7:10])), dim=1)
                tot_ball_vel_rew += 1.0/(1. + ball_vel)
            return tot_ball_vel_rew
        else:
            # 相对机器人的速度
            ball_vel = torch.norm(quat_rotate_inverse(self.base_quat,(self.root_states[:, 1, 7:10]- 
                                                                    self.root_states[:, 0, 7:10])), dim=1)
            tot_ball_vel_rew += 1.0/(1. + ball_vel)
            return tot_ball_vel_rew


    # def _reward_ball_ang_vel(self):
    #     """ Penalize the ball's angular velocity """
    #     ball_ang_vel = torch.norm(self.ball_angvels, dim=1)
    #     return torch.square(ball_ang_vel)
    
    def _reward_ball_dist(self):
        """ Penalize the ball's distance from the robot """
        tot_ball_dist_rew = 0
        if self.multi_balls:
            for i in range(self.num_balls):
                ball_dist = torch.norm(self.ball_positions[:,i,:] - self.root_states[:, 0, :3], dim=1)
                tot_ball_dist_rew += 1.0/(1.0 + ball_dist)
            return tot_ball_dist_rew

        else:
            ball_dist = torch.norm(self.root_states[:,1,0:3] - self.root_states[:, 0, :3], dim=1)
            tot_ball_dist_rew += 1.0/(1.0 + ball_dist)
            return tot_ball_dist_rew

    ############# load wrenches related rewards #############
    # def _reward_wrench_smoothness(self):
    #     """ Penalize the wrench smoothness """
    #     return torch.sum(torch.square(self.wrenches_on_robot - self.last_wrenches_on_robot), dim=1)

    def _reward_power(self):
        # Penalize power
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - 2.* self.last_actions - self.last_last_actions), dim=1)
