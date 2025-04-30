import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, to_torch, get_axis_params, quat_mul, quat_apply
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_float
from legged_gym.utils.helpers import class_to_dict
from .pointfoot_flat import BipedPF
import os
from legged_gym import LEGGED_GYM_ROOT_DIR

class BipedPFBallBalance(BipedPF):  # Now inherits from BipedPF directly
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.num_balls = cfg.env.num_balls
        self.num_actors = 1 + self.num_balls
        cfg.env.num_actors = self.num_actors # Ensure config reflects reality if parent uses it
        
        # Let parent initialize fully. It will call our overridden _create_envs and _init_buffers.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # --- Parent initialization complete ---
        # self.num_envs, self.device, self.envs[] are now set.
        print("command_ranges in __init__:", self.command_ranges)
        # Optional: Viewer setup
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, loads ball asset
             2. Initializes ball properties (mass, radius)
             3. For each environment
                3.1 creates the environment,
                3.2 calls DOF and Rigid shape properties callbacks for robot,
                3.3 creates robot actor,
                3.4 creates ball actor(s) with their properties.
             4. Store indices of different bodies of the robot
        """
        # --- Initialize Buffers needed by _process_..._props methods --- 
        # These are normally initialized in _init_buffers, but are needed earlier here.
        self.base_com = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Also potentially friction_coeffs and restitution_coef if needed by _process_rigid_shape_props
        # Initialize them here if the parent _process_rigid_shape_props uses them before setting them.
        # self.friction_coeffs = torch.zeros(self.num_envs, 1, device='cpu') # Check parent code for device
        # self.restitution_coef = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # --- Initialize Ball Buffers Needed for Actor Creation ---
        # We need mass and radius *before* the loop.
        # Call _init_ball_buffers here, but it might need self.device, self.num_envs?
        # Let's initialize directly here for now.
        print("Initializing ball mass/radius arrays before env creation loop...")
        self.ball_radius = torch_rand_float(
            self.cfg.ball.radius_range[0],
            self.cfg.ball.radius_range[1],
            (self.num_envs, self.num_balls),
            device=self.device # self.device should be set by BaseTask.__init__ before this
        )
        self.ball_mass = torch_rand_float(
            self.cfg.ball.mass_range[0],
            self.cfg.ball.mass_range[1],
            (self.num_envs, self.num_balls),
            device=self.device
        )
        print("Ball mass/radius arrays initialized.")

        # --- Load Robot Asset ---
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR = LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        # ... (copy all asset_options settings from BaseTask._create_envs) ...
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

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies_robot = len(body_names) # Store robot-specific count if needed
        self.num_dofs = len(self.dof_names)

        # --- Prepare Initial State (Copied from BaseTask) ---
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        ) # Still 1D

        # --- Get Env Origins (Call it HERE, before the loop) ---
        self._get_env_origins()
        # Ensure contiguous memory for safety
        self.env_origins = self.env_origins.contiguous()
        print(f"_create_envs: Created env_origins with shape: {self.env_origins.shape}")

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)

        # --- Environment Creation Loop ---
        self.actor_handles = []
        self.envs = []
        self.ball_handles = [] # Initialize list for ball handles

        for i in range(self.num_envs):
            # Create env
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            self.envs.append(env_handle)

            # Set start pose for robot
            start_pose = gymapi.Transform()
            pos = self.env_origins[i].clone() # Use the correct origin for this env
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1) # Random offset
            start_pose.p = gymapi.Vec3(*pos)

            # Create Robot Actor
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i) # Parent method call
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i, # Collision group
                self.cfg.asset.self_collisions,
                0 # Collision filter
            )
            self.actor_handles.append(robot_handle) # Add robot handle

            # Process DOF and Body Props for Robot
            dof_props = self._process_dof_props(dof_props_asset, i) # Parent method call
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, i) # Parent method call
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)

            # Create Ball Actor(s) for this environment
            env_ball_handles = []
            for ball_idx in range(self.num_balls):
                ball_options = gymapi.AssetOptions()
                # Density calculation needs mass and radius for *this* env/ball
                ball_options.density = self.ball_mass[i, ball_idx] / (
                    (4/3) * np.pi * self.ball_radius[i, ball_idx]**3 + 1e-6 # Add epsilon for safety
                )
                ball_asset = self.gym.create_sphere(
                    self.sim,
                    self.ball_radius[i, ball_idx].item(),
                    ball_options
                )
                # Set ball properties (friction etc.) if needed by processing shape props
                # ball_shape_props = self.gym.get_asset_rigid_shape_properties(ball_asset)
                # processed_ball_props = self._process_ball_rigid_shape_props(ball_shape_props, i) # Need this method
                # self.gym.set_asset_rigid_shape_properties(ball_asset, processed_ball_props)

                ball_pose = gymapi.Transform()
                ball_initial_pos = self._get_initial_ball_position(i, ball_idx) # Relative to robot start? Check this.
                # If relative, add robot start pos: ball_pose.p = start_pose.p + gymapi.Vec3(*ball_initial_pos)
                # If absolute in env: ball_pose.p = gymapi.Vec3(*ball_initial_pos) + self.env_origins[i]? Needs care.
                # Assuming _get_initial_ball_position provides world coords relative to env origin for now:
                #print("self.env_origins[i] in _create_envs:", self.env_origins[i])
                ball_world_pos = self.env_origins[i] + to_torch(ball_initial_pos, device=self.device)
                ball_pose.p = gymapi.Vec3(*ball_world_pos)

                ball_handle = self.gym.create_actor(
                    env_handle,
                    ball_asset,
                    ball_pose,
                    f"ball_{ball_idx}",
                    i, # Collision group
                    0, # Self collision (e.g., 0 for collide with others)
                    0 # Collision filter
                )
                env_ball_handles.append(ball_handle)
            self.ball_handles.append(env_ball_handles) # Store handles for this env

        # --- Post-Loop Setup --- 
        # Update self.num_bodies to the ACTUAL total number of bodies per env
        # BEFORE parent's _init_buffers uses it.
        if self.num_envs > 0:
            # Get count from the first initialized environment
            self.num_bodies = self.gym.get_env_rigid_body_count(self.envs[0])
            print(f"_create_envs: Updated self.num_bodies to {self.num_bodies}")
        else:
            self.num_bodies = 0 # Or handle appropriately

        # --- Feet indices etc. (Copied from BaseTask) ---
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            # Use robot handle (index 0 in actor_handles)
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _init_ball_buffers(self):
        """Initialize buffers specifically for ball tracking. Assumes parent init is done."""
        # Ball properties (safe to use self.num_envs, self.device)
        self.ball_radius = torch_rand_float(
            self.cfg.ball.radius_range[0], 
            self.cfg.ball.radius_range[1],
            (self.num_envs, self.num_balls),
            device=self.device # Use device set by parent
        )
        self.ball_mass = torch_rand_float(
            self.cfg.ball.mass_range[0],
            self.cfg.ball.mass_range[1],
            (self.num_envs, self.num_balls),
            device=self.device
        )
        
        # Wrench tracking
        self.wrenches_on_robot = torch.zeros(self.num_envs, 6, device=self.device)
        self.wrench_buffer = torch.zeros(self.num_envs, 50, 6, device=self.device)
        
        # Ball state references (point to slices in the *resized* self.root_states)
        self.ball_positions = self.root_states[:, 1:, :3]
        self.ball_linvels = self.root_states[:, 1:, 7:10]

    def _add_balls_to_envs(self):
        """Adds ball actors to the environments already created by the parent."""
        # Assumes self.envs, self.sim, self.num_envs, self.ball_mass, self.ball_radius exist
        self.ball_handles = []
        for env_idx in range(self.num_envs):
            env_handles = []
            for ball_idx in range(self.num_balls):
                ball_options = gymapi.AssetOptions()
                # Ensure attributes exist (Good practice, though order should be guaranteed now)
                if not hasattr(self, 'ball_mass') or not hasattr(self, 'ball_radius'):
                     raise RuntimeError("Ball properties not initialized before _add_balls_to_envs")

                ball_options.density = self.ball_mass[env_idx, ball_idx] / (
                    (4/3) * np.pi * self.ball_radius[env_idx, ball_idx]**3
                )
                ball_asset = self.gym.create_sphere(
                    self.sim, 
                    self.ball_radius[env_idx, ball_idx].item(),
                    ball_options
                )
                ball_pose = gymapi.Transform()
                ball_pose.p = gymapi.Vec3(*self._get_initial_ball_position(env_idx, ball_idx))
                
                ball_handle = self.gym.create_actor(
                    self.envs[env_idx], # Use existing env handle from parent
                    ball_asset,
                    ball_pose,
                    f"ball_{ball_idx}",
                    env_idx, 0, 0 # Collision group/filter
                )
                env_handles.append(ball_handle)
            self.ball_handles.append(env_handles)

    def _get_initial_ball_position(self, env_idx, ball_idx):
        """Position balls in a configurable pattern around robot"""
        if self.cfg.ball.spawn_pattern == "circle":
            angle = 2 * np.pi * ball_idx / self.num_balls
            radius = self.cfg.ball.spawn_radius
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
        elif self.cfg.ball.spawn_pattern == "random":
            x = torch_rand_float(-1, 1, (1,), device=self.device).item() * self.cfg.ball.spawn_radius
            y = torch_rand_float(-1, 1, (1,), device=self.device).item() * self.cfg.ball.spawn_radius
        else:  # "fixed"
            x = y = 0
            
        z = self.ball_radius[env_idx, ball_idx].item() + 0.1
        return [x, y, z]

    def post_physics_step(self):
        """Overrides BaseTask.post_physics_step to handle multi-actor root_states.
        Checks terminations, computes observations and rewards.
        Calls self._post_physics_step_callback() for common computations.
        Calls self._draw_debug_vis() if needed.
        """
        # --- Refresh tensors --- 
        # (These are critical and should be called first) 
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) # Need DOF state for torque calc etc.
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # --- Update Ball States FIRST --- 
        # This updates self.root_states with the correct multi-actor shape
        self._update_ball_states()
        
        # --- Base class logic adapted for multi-actor --- 
        self.episode_length_buf += 1

        # Prepare base robot quantities (use actor index 0)
        # Ensure self.root_states has the correct shape before indexing
        if self.root_states.shape[1] < 1:
             # Handle error: root_states doesn't have the expected actor dimension
             raise ValueError(f"post_physics_step: Invalid root_states shape {self.root_states.shape}, expected at least 1 actor.")

        self.base_quat[:] = self.root_states[:, 0, 3:7] # Select robot actor 0
        self.base_position = self.root_states[:, 0, :3] # ADDED: Explicitly extract robot position
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10]) # Select robot actor 0
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13]) # Select robot actor 0
        
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt # Assuming dof_vel is updated correctly elsewhere
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt # Assuming dof_pos is updated correctly elsewhere
        self.power = torch.abs(self.torques * self.dof_vel) # Assumes torques/dof_vel are correct

        self.compute_foot_state() # Assumes this uses updated rigid_body_state or foot_positions
        
        # --- Ball-Specific Logic --- 
        self._compute_wrenches() # Calculate ball interaction forces
        # NOTE: Reward calculation is moved down, after resets

        # --- Callbacks (As in BaseTask) --- 
        # This method might also need adapting if it accesses root_states incorrectly
        if hasattr(self, '_post_physics_step_callback'):
             self._post_physics_step_callback()

        # --- Compute Observations, Rewards, Resets --- 
        self.check_termination() # Child version (includes ball checks)
        
        # Combine rewards (locomotion + ball) using child's logic
        self._combine_rewards() # This calculates and sets self.rew_buf
        
        # Handle resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
             self.reset_idx(env_ids) # Child's reset_idx handles balls and obs history
             # print("[DEBUG post_physics_step] Resetting envs:", env_ids) # Optional Debug
        
        # Compute final observations (Child version handles robot+balls)
        self.compute_observations()

        # --- Update History Buffers (As in BaseTask) --- 
        # Make sure slices match tensor dimensions
        # last_actions needs care if it has extra dim? Check base task init
        if self.last_actions.dim() == 3 and self.last_actions.shape[2] == 2:
             # Assuming shape is [N, num_actions, 2] for history
             self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
             self.last_actions[:, :, 0] = self.actions[:]
        elif self.last_actions.dim() == 2:
             # If shape is [N, num_actions] (no history dim)
             self.last_actions[:] = self.actions[:]
        else:
             print(f"Warning: Unexpected last_actions shape {self.last_actions.shape}")

        self.last_dof_vel[:] = self.dof_vel[:] # Assuming dof_vel is correct
        
        # last_root_vel needs to come from robot actor 0
        self.last_root_vel[:] = self.root_states[:, 0, 7:13] # Select robot actor 0
        
        # Update base_position from robot state if needed, requires extracting it first
        self.base_position = self.root_states[:, 0, :3] # Ensure this is updated if needed
        self.last_base_position[:] = self.base_position[:]
        
        # Update foot positions if they exist
        if hasattr(self, 'foot_positions'): # Check attribute exists
             self.last_foot_positions[:] = self.foot_positions[:]

    def _update_ball_states(self):
        """Refresh ball positions/velocities from the simulation"""
        # Get all actor states (robot + balls) from the simulation
        # Use acquire_... and refresh is done separately in post_physics_step
        all_states_flat = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        
        # --- CORRECTED LOGIC --- 
        # Reshape the flat tensor to the multi-actor format FIRST
        # Ensure self.num_actors is correct (should be 1 + num_balls)
        if all_states_flat.shape[0] != self.num_envs * self.num_actors:
             # This indicates a potential mismatch, maybe num_actors wasn't updated?
             print(f"Warning in _update_ball_states: Flat state shape {all_states_flat.shape} inconsistent with num_envs={self.num_envs}, num_actors={self.num_actors}")
             # Attempt to proceed, but this might indicate deeper issues
        
        # Reshape based on expected num_actors
        try:
            all_states_reshaped = all_states_flat.view(self.num_envs, self.num_actors, 13)
        except RuntimeError as e:
            print(f"Error reshaping flat state tensor: {e}")
            print(f"  Flat shape: {all_states_flat.shape}, Target shape: ({self.num_envs}, {self.num_actors}, 13)")
            # Don't attempt update if reshape fails
            return 
            
        # Update the entire self.root_states with the newly fetched and reshaped data
        # This ensures self.root_states always has the correct, updated values and shape
        self.root_states[:] = all_states_reshaped
        
        # Note: ball_positions and ball_linvels are views defined in _init_buffers,
        # they automatically reflect the updated self.root_states.
        # No need to re-assign them here.

    def _compute_wrenches(self):
        """Calculate interaction forces between robot and all balls"""
        self.wrenches_on_robot[:] = 0
        
        for ball_idx in range(self.num_balls):
            # Get contact forces for this ball across all envs
            contact_forces = torch.stack([
                self.gym.get_actor_contact_forces(
                    self.sim, env, self.ball_handles[env_idx][ball_idx])
                for env_idx, env in enumerate(self.envs)
            ])
            
            # Transform to robot frame and accumulate
            forces = quat_rotate_inverse(self.base_quat, contact_forces)
            moments = torch.cross(
                self.ball_position[:, ball_idx] - self.base_position, 
                forces
            )
            self.wrenches_on_robot += torch.cat([forces, moments], dim=-1)
        
        # Update wrench history
        self.wrench_buffer = torch.roll(self.wrench_buffer, shifts=1, dims=1)
        self.wrench_buffer[:, 0] = self.wrenches_on_robot

    def _combine_rewards(self):
        """Blend locomotion and balancing rewards"""
        loco_reward = 0.6 * self._reward_tracking_lin_vel() 
        bal_reward = 0.4 * self._reward_ball_balance()
        self.rew_buf = loco_reward + bal_reward

    def compute_observations(self):
        """ Computes observations for the robot AND the ball(s),
            and handles noise addition correctly for the combined observation space.
            Overrides the parent method entirely.
        """
        # 1. Compute base robot observations by calling the parent's method
        #    BipedPF provides compute_group_observations which returns (obs, critic_obs)
        if hasattr(self, 'compute_group_observations'):
            # Get the tuple (robot_obs, robot_critic_obs) from the parent
            robot_obs, robot_critic_obs = self.compute_group_observations() # Shape [N, 30], [N, CritObsSize]
        else:
            # This should not happen if inheriting correctly from BipedPF
            print("CRITICAL Error: compute_group_observations not found.")
            raise AttributeError("Cannot compute base observations, compute_group_observations is missing.")

        # 2. Calculate ball observations (using the reshaping logic from previous step)
        robot_pos_slice = self.root_states[:, 0, :3]
        robot_lin_vel_slice = self.root_states[:, 0, 7:10]
        base_quat = self.base_quat

        if self.num_balls > 0:
            relative_ball_pos_3d = self.ball_positions - robot_pos_slice.unsqueeze(1)
            relative_ball_vel_3d = self.ball_linvels - robot_lin_vel_slice.unsqueeze(1)
            expanded_base_quat = base_quat.unsqueeze(1).expand(-1, self.num_balls, -1)
            quat_flat = expanded_base_quat.reshape(-1, 4)
            relative_pos_flat = relative_ball_pos_3d.reshape(-1, 3)
            relative_vel_flat = relative_ball_vel_3d.reshape(-1, 3)
            rotated_pos_flat = quat_rotate_inverse(quat_flat, relative_pos_flat)
            rotated_vel_flat = quat_rotate_inverse(quat_flat, relative_vel_flat)
            rotated_pos_obs = rotated_pos_flat.reshape(self.num_envs, -1)
            rotated_vel_obs = rotated_vel_flat.reshape(self.num_envs, -1)
            ball_mass_obs = self.ball_mass.reshape(self.num_envs, -1)
            ball_obs = torch.cat([rotated_pos_obs, rotated_vel_obs, ball_mass_obs], dim=-1)
        else:
            ball_obs = torch.empty((self.num_envs, 0), device=self.device)

        # 3. Concatenate robot and ball observations for the main obs buffer
        self.obs_buf = torch.cat([robot_obs, ball_obs], dim=-1) # Final shape [N, 30 + M*7]

        # 4. Apply Noise to the combined obs_buf
        if self.add_noise:
            # Ensure noise_scale_vec matches the *combined* obs_buf size
            if hasattr(self, 'noise_scale_vec') and self.noise_scale_vec.shape[0] == self.obs_buf.shape[1]:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec.unsqueeze(0)
            elif hasattr(self, 'noise_scale_vec'):
                print(f"Warning: Noise scale vec shape {self.noise_scale_vec.shape} doesn't match obs_buf dim 1 {self.obs_buf.shape[1]}. Noise not applied correctly.")
            else:
                print("Warning: add_noise is True but noise_scale_vec not found.")

        # 5. Update Observation History Buffer
        if hasattr(self, 'obs_history_buf') and hasattr(self, 'obs_history_length'):
            if self.obs_history_buf is not None and self.obs_history_buf.shape[1] == self.obs_buf.shape[1] * self.obs_history_length:
                 self.obs_history_buf = torch.cat((self.obs_history_buf[:, self.obs_buf.shape[1]:], self.obs_buf), dim=1)
            else:
                 print(f"Warning: obs_history_buf shape mismatch or not initialized. Shape: {getattr(self.obs_history_buf, 'shape', 'None')}, Expected cols: {self.obs_buf.shape[1] * self.obs_history_length}")

        # --- Print shape for confirmation ---
        # print(f"*** BipedPFBallBalance: Final computed obs_buf shape: {self.obs_buf.shape} ***")

        # 6. Handle Privileged Observations / Critic Observations
        #    The parent returned robot_critic_obs. We likely need to add ball info
        #    if the critic needs it. For now, just pass it through or handle as needed.
        #    Example: Concatenate ball obs also to critic obs
        if hasattr(self, 'critic_obs_buf'): # Check if critic buffer exists
             # Decide how to combine: maybe critic needs same extension?
             self.critic_obs_buf = torch.cat([robot_critic_obs, ball_obs], dim=-1) # Example combination
             # Or maybe critic only needs parent obs: self.critic_obs_buf = robot_critic_obs
        else:
             # Handle case where critic buffer isn't used/initialized
             print("Warning: critic_obs_buf not found.")
             # If parent returned it, maybe we should too, even if unused internally?
             # robot_critic_obs = None # Or keep the value from parent call

        # 7. Return value(s) - Match parent's structure (obs, critic_obs)
        #    Return the combined obs_buf and the (potentially combined) critic_obs_buf
        if hasattr(self, 'critic_obs_buf'):
             return self.obs_buf, self.critic_obs_buf
        else:
             # If parent returned tuple but we don't have critic_obs, return placeholder?
             return self.obs_buf, robot_critic_obs # Return what parent gave for critic obs

    def check_termination(self):
        # Parent checks for locomotion failures
        super().check_termination()
        
        # Additional checks for ball conditions
        ball_dist = torch.norm(
            self.ball_positions[:, :, :2] - self.base_position[:, :2].unsqueeze(1), # Corrected: base_position (singular)
            dim=-1
        )
        self.reset_buf |= (ball_dist > self.cfg.ball.max_distance).any(dim=1)

    def reset_idx(self, env_ids):
        """ Resets specified environments.
            Overrides parent to handle extended observation space correctly,
            especially for the observation history buffer.
        """
        print("--------------------------------")
        print("env_ids.shape in reset_idx:", env_ids.shape)
        print("env_ids in reset_idx:", env_ids)
        print("--------------------------------")
        if len(env_ids) == 0:
            return

        # Debug prints for envs_steps_buf before reset
        print(f"[DEBUG Reset] Before reset - envs_steps_buf shape: {self.envs_steps_buf.shape}")
        print(f"[DEBUG Reset] Before reset - envs_steps_buf device: {self.envs_steps_buf.device}")
        print(f"[DEBUG Reset] Before reset - envs_steps_buf[0]: {self.envs_steps_buf[0]}")

        # --- 1. Perform necessary resets usually done by parent BipedPF/BaseTask ---
        # Call underlying methods directly to replicate essential parent logic.
        # Ensure these methods exist (inherited or defined).

        if hasattr(self, '_reset_dofs'):
            self._reset_dofs(env_ids)
        else:
            print("Warning: _reset_dofs method not found.")
        print("--------------------------------")
        print("env_ids in reset_idx, after _reset_dofs:", env_ids)
        print("--------------------------------")
        if hasattr(self, '_reset_root_states'):
            # This uses the _reset_root_states we specifically overrode
            # to handle 1D base_init_state and 3D root_states.
            self._reset_root_states(env_ids)
        else:
            print("Warning: _reset_root_states method not found.")
        # ADDED: Reset last_base_position after root states are reset
        if hasattr(self, 'root_states') and hasattr(self, 'last_base_position'):
             self.last_base_position[env_ids] = self.root_states[env_ids, 0, :3]
        elif not hasattr(self, 'last_base_position'):
             print("Warning: last_base_position not found during reset.")

        print("command_ranges in reset_idx, before _resample_commands:", self.command_ranges)
        #print("shape of command_ranges in reset_idx, before _resample_commands:", self.command_ranges["lin_vel_x"].shape)
        if hasattr(self, '_resample_commands'):
            self._resample_commands(env_ids)
        else:
            print("Warning: _resample_commands method not found.")

        # Reset common buffers (check BipedPF/BaseTask reset_idx for others if needed)
        if hasattr(self, 'last_actions'):
            self.last_actions[env_ids] = 0.
        if hasattr(self, 'last_dof_vel'):
            self.last_dof_vel[env_ids] = 0.
        if hasattr(self, 'episode_length_buf'):
            self.episode_length_buf[env_ids] = 0
        if hasattr(self, 'reset_buf'):
            self.reset_buf[env_ids] = 1
        # Reset any other essential parent buffers here...
        if hasattr(self, 'dof_pos_int'): # ADDED: Reset DOF integral
             self.dof_pos_int[env_ids] = 0.

        # Reset envs_steps_buf
        if hasattr(self, 'envs_steps_buf'):
            self.envs_steps_buf[env_ids] = 0
            # Debug prints for envs_steps_buf after reset
            print(f"[DEBUG Reset] After reset - envs_steps_buf shape: {self.envs_steps_buf.shape}")
            print(f"[DEBUG Reset] After reset - envs_steps_buf device: {self.envs_steps_buf.device}")
            print(f"[DEBUG Reset] After reset - envs_steps_buf[0]: {self.envs_steps_buf[0]}")
        else:
            print("Warning: envs_steps_buf not found during reset!")

        # --- 2. Child-Specific Resets (Balls) ---
        for env_id in env_ids:
            for ball_idx in range(self.num_balls):
                pos = self._get_initial_ball_position(env_id, ball_idx)
                # Initialize full state (pos, rot, lin_vel, ang_vel) for the ball
                vel = [0, 0, 0]
                quat = [0, 0, 0, 1] # Default orientation
                ang_vel = [0, 0, 0]
                ball_state = torch.tensor(pos + quat + vel + ang_vel, device=self.device)
                # Ensure self.root_states has the actor dimension
                if self.root_states.dim() == 3 and self.root_states.shape[1] > 1 + ball_idx:
                     self.root_states[env_id, 1 + ball_idx, :] = ball_state
                else:
                     print(f"Error/Warning: Problem assigning ball state. root_states shape: {self.root_states.shape}, ball_idx: {ball_idx}")

        # --- 3. Compute FULL observations AFTER states are reset ---
        # This calculates self.obs_buf using the child's logic (robot + balls)
        self.compute_observations()
        # print(f"Computed obs_buf shape in reset_idx: {self.obs_buf.shape}") # Optional DEBUG PRINT

        # --- 4. Correctly reset observation history buffer ---
        if hasattr(self, 'obs_history_length') and hasattr(self, 'obs_history_buf'):
             # Ensure dimensions match before assignment
             expected_history_cols = self.obs_buf.shape[1] * self.obs_history_length
             if self.obs_history_buf.shape[1] != expected_history_cols:
                  # Handle potential mismatch if buffer wasn't initialized correctly for child obs size
                  print(f"Warning: Resizing obs_history_buf in reset. Old shape: {self.obs_history_buf.shape}, New expected shape: {(self.num_envs, expected_history_cols)}")
                  self.obs_history_buf = torch.zeros(self.num_envs, expected_history_cols, dtype=torch.float, device=self.device)

             # Perform the reset using the correctly shaped obs_buf
             self.obs_history_buf[env_ids] = self.obs_buf[env_ids].repeat(1, self.obs_history_length)
        else:
             # Try to infer history length if attribute is missing
             if hasattr(self, 'obs_history_buf') and self.obs_buf.shape[1] > 0:
                 inferred_history_length = self.obs_history_buf.shape[1] // self.obs_buf.shape[1]
                 if inferred_history_length > 0:
                     print(f"Warning: obs_history_length not found, inferring as {inferred_history_length}. Resetting history buffer.")
                     self.obs_history_buf[env_ids] = self.obs_buf[env_ids].repeat(1, inferred_history_length)
                 else:
                     print("Warning: Could not infer obs_history_length. Cannot reset history buffer.")
             else:
                print("Warning: obs_history_length or obs_history_buf not found/usable. Cannot reset history buffer.")

        # --- DEBUG CHECKS before setting actor state ---
        print(f"[DEBUG Reset Idx] root_states shape: {self.root_states.shape}, device: {self.root_states.device}, is_contig: {self.root_states.is_contiguous()}")
        if torch.isnan(self.root_states).any():
            print("[DEBUG Reset Idx] CRITICAL: NaNs found in root_states before setting!")
        else:
            print("[DEBUG Reset Idx] No NaNs found in root_states.")

        actor_indices = None
        if hasattr(self, 'all_actor_indices'):
            try:
                 actor_indices = self.all_actor_indices[env_ids].flatten()
                 print(f"[DEBUG Reset Idx] actor_indices shape: {actor_indices.shape}, device: {actor_indices.device}, min/max: {actor_indices.min()}/{actor_indices.max()}")
                 # Check bounds
                 max_expected_index = self.num_envs * self.num_actors -1
                 if actor_indices.max() > max_expected_index or actor_indices.min() < 0:
                      print(f"[DEBUG Reset Idx] CRITICAL: actor_indices out of bounds! Min/Max: {actor_indices.min()}/{actor_indices.max()}, Expected Range: 0-{max_expected_index}")
            except IndexError as e:
                 print(f"[DEBUG Reset Idx] Error accessing all_actor_indices: {e}")
                 actor_indices = None # Fallback to non-indexed
        else:
            print("[DEBUG Reset Idx] self.all_actor_indices not found.")
        # --- END DEBUG CHECKS ---

        # --- 6. Final refresh of simulation state tensor ---
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # Use indexed update if indices are valid
        if actor_indices is not None and hasattr(self, 'all_actor_indices'):
             print("[DEBUG Reset Idx] Using set_actor_root_state_tensor_indexed")
             self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                          gymtorch.unwrap_tensor(self.root_states),
                                                          gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        else: # Fallback to full update
             print("[DEBUG Reset Idx] Using set_actor_root_state_tensor (Full)")
             self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        print("[DEBUG Reset Idx] set_actor_root_state_tensor call completed.")

        # --- 7. Handle Logging ---
        # Replicate necessary logging from parent/base class. Check BaseTask.reset_idx
        if hasattr(self, 'extras') and hasattr(self, 'episode_sums'):
             self.extras["episode"] = {}
             for key in self.episode_sums.keys():
                 # Ensure attributes used exist
                 max_len_s = getattr(self, 'max_episode_length_s', 1.0) # Default to 1 if not found
                 if key in self.episode_sums: # Check key exists before accessing
                      self.extras["episode"]['rew_' + key] = torch.mean(
                          self.episode_sums[key][env_ids]) / max_len_s
                      self.episode_sums[key][env_ids] = 0.
             # Add curriculum/timeout logging if applicable and attributes exist
             # Use getattr for safer access to potentially missing config attributes
             if getattr(self.cfg.terrain, 'curriculum', False) and hasattr(self, 'terrain_levels'):
                  self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels[env_ids].float()) # Use env_ids subset? Check BaseTask
             if getattr(self.cfg.commands, 'curriculum', False) and hasattr(self, 'command_ranges'):
                print("command_ranges in reset_idx:", self.command_ranges)
                print("shape of command_ranges in reset_idx:", self.command_ranges["lin_vel_x"].shape)
                self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
             if getattr(self.cfg.env, 'send_timeouts', False) and hasattr(self, 'time_out_buf'):
                  self.extras["time_outs"] = self.time_out_buf[env_ids] # Ensure time_out_buf exists
        print("--------------------------------")
        print("env_ids in reset_idx, at the end of reset_idx:", env_ids)
        print("--------------------------------")

    # Enhanced reward functions ==============================================

    def _reward_ball_balance(self):
        """Reward for keeping balls near desired positions"""
        # Target positions relative to robot (can be configured)
        target_pos = torch.zeros(self.num_envs, self.num_balls, 2, device=self.device)
        
        # Calculate position errors
        pos_errors = torch.norm(
            self.ball_positions[:, :, :2] - (self.base_position[:, :2].unsqueeze(1) + target_pos),
            dim=-1
        )
        
        # Calculate velocity penalties
        vel_penalties = torch.norm(self.ball_linvels[:, :, :2], dim=-1)
        
        # Combine with weights
        return torch.exp(
            -self.cfg.ball.pos_weight * pos_errors.mean(dim=1) 
            -self.cfg.ball.vel_weight * vel_penalties.mean(dim=1)
        )

    def _reward_wrench_smoothness(self):
        """Penalize abrupt changes in interaction forces"""
        # Check if envs_steps_buf exists and has data before taking mean
        if not hasattr(self, 'envs_steps_buf') or self.envs_steps_buf.numel() == 0:
            return torch.zeros_like(self.rew_buf) # Return zero if buffer isn't ready
            
        # Corrected attribute name: envs_steps_buf
        if self.envs_steps_buf.mean() < 10:  # Skip initial steps
            return torch.zeros_like(self.rew_buf)

    # Maintain all original BipedPF locomotion rewards ======================
    # All parent reward functions remain available:
    # _reward_tracking_lin_vel()
    # _reward_tracking_ang_vel() 
    # _reward_feet_air_time()
    # etc...

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocity to initial states for selected envs
            Handles the 3D root_states tensor (robot + balls) and 1D base_init_state.
            Accounts for 2D env_origins in the plane case.
            Args:
                env_ids (torch.Tensor or List[int]): Environment ids
        """
        # --- Add these checks right at the beginning ---
        print(f"[DEBUG Reset Root States START] env_ids shape: {env_ids.shape}, device: {env_ids.device}")
        if hasattr(self, 'env_origins'):
            print(f"[DEBUG Reset Root States START] env_origins shape: {self.env_origins.shape}, device: {self.env_origins.device}, is_contig: {self.env_origins.is_contiguous()}")
            # Try accessing a single element
            try:
                print(f"[DEBUG Reset Root States START] env_origins[0]: {self.env_origins[0]}")
            except Exception as e:
                print(f"[DEBUG Reset Root States START] ERROR accessing env_origins[0]: {e}")
        else:
            print("[DEBUG Reset Root States START] env_origins attribute NOT FOUND!")
        # --- End checks ---

        if len(env_ids) == 0:
            return

        # Ensure base_init_state exists and has the expected 1D shape
        if not hasattr(self, 'base_init_state'):
            raise AttributeError("self.base_init_state not found during reset.")
        if self.base_init_state.dim() != 1 or self.base_init_state.shape[0] != 13:
             raise ValueError(f"Unexpected shape for base_init_state: {self.base_init_state.shape}. Expected 1D (13,).")

        num_envs_to_reset = len(env_ids)

        # Extract the template state components
        base_pos = self.base_init_state[:3]
        base_rot = self.base_init_state[3:7]
        base_lin_vel = self.base_init_state[7:10]
        base_ang_vel = self.base_init_state[10:13]

        # Expand the single template state to match the number of envs being reset
        expanded_pos = base_pos.unsqueeze(0).expand(num_envs_to_reset, -1)
        expanded_rot = base_rot.unsqueeze(0).expand(num_envs_to_reset, -1)
        expanded_lin_vel = base_lin_vel.unsqueeze(0).expand(num_envs_to_reset, -1)
        expanded_ang_vel = base_ang_vel.unsqueeze(0).expand(num_envs_to_reset, -1)

        # --- Assign expanded state ONLY to the robot's slice (actor index 0) ---
        # Position (apply env origin offset only to robot)
        if self.custom_origins: # Terrain case: env_origins is likely 3D (num_envs, num_actors, 3)
             self.root_states[env_ids, 0, :3] = expanded_pos + self.env_origins[env_ids, 0, :3] # Indexing [env_ids, 0, :3] is correct here
        else: # Plane case: env_origins is 2D (num_envs, 3)
             # Use correct 2D indexing for env_origins
             print(f"env_origins shape: {self.env_origins.shape}")
             print(f"expanded_pos shape: {expanded_pos.shape}")
             print("self.env_origins[env_ids]:", self.env_origins[env_ids])
             print("expanded_pos + self.env_origins[env_ids]:", expanded_pos + self.env_origins[env_ids])
             print("self.root_states[env_ids, 0, :3]:", self.root_states[env_ids, 0, :3])
             self.root_states[env_ids, 0, :3] = expanded_pos + self.env_origins[env_ids, :] # <--- FIX: Use [env_ids, :]

        # Rotation
        self.root_states[env_ids, 0, 3:7] = expanded_rot

        # Linear Velocity (add randomization)
        rand_lin_vel = torch_rand_float(-0.5, 0.5, (num_envs_to_reset, 3), device=self.device)
        self.root_states[env_ids, 0, 7:10] = expanded_lin_vel # + rand_lin_vel # Optional randomization

        # Angular Velocity (add randomization)
        rand_ang_vel = torch_rand_float(-0.5, 0.5, (num_envs_to_reset, 3), device=self.device)
        self.root_states[env_ids, 0, 10:13] = expanded_ang_vel # + rand_ang_vel # Optional randomization

        # Ball states are handled in the main reset_idx method
        # The final set_actor_root_state_tensor is also handled there

    def _init_buffers(self):
        """Initialize torch tensors. Handles all necessary initializations explicitly, adapted for multi-actor."""
        print("--------------------------------")
        print("BipedPFBallBalance: Initializing buffers...")
        # print("command_ranges at start of _init_buffers:", self.command_ranges) # Debug if needed
        print("--------------------------------")

        # --- 1. Acquire Gym GPU state tensors ---
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim) # Added
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)   # Added early

        # --- 2. Refresh tensors --- Added
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # --- 3. Create wrappers and reshape for multi-actor/multi-body ---
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, self.num_actors, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13) # Reshape depends on num_bodies being correct

        # Contact forces (Added)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces_tensor).view(
            self.num_envs, -1, 3 # Shape: num_envs, num_bodies (or contacts?), xyz axis
        )

        # --- 4. Initialize common views and buffers ---
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_acc = torch.zeros_like(self.dof_vel)

        # Robot-specific views (using actor index 0)
        self.base_quat = self.root_states[:, 0, 3:7]
        self.base_position = self.root_states[:, 0, :3] # Added early view
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.base_height = torch.zeros_like(self.root_states[:, 0, 2]) # Added

        # Common physics vectors
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # --- 5. Initialize Control Buffers ---
        self.power = torch.zeros( # Added
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # History buffers
        self.last_actions = torch.zeros( # Corrected shape
            self.num_envs,
            self.num_actions,
            2, # Added history dimension
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 0, 7:13]) # Added, referencing robot slice
        self.dof_pos_int = torch.zeros_like(self.dof_pos)
        self.last_base_position = self.base_position.clone() # Use clone of early view

        # --- 6. Initialize Action Delay/FIFO Buffers ---
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max_sec = self.cfg.domain_rand.delay_ms_range[1] / 1000.0
        delay_max_steps = np.ceil(delay_max_sec / self.sim_params.dt)
        delay_max = np.int64(delay_max_steps)
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # Apply randomization if configured
        if self.cfg.domain_rand.randomize_action_delay: # Correction: Renamed from randomize_action_delay_idx
            min_delay_sec = self.cfg.domain_rand.delay_ms_range[0] / 1000.0
            min_delay_steps = np.floor(min_delay_sec / self.sim_params.dt)
            action_delay_steps = torch_rand_float(
                min_delay_steps, delay_max_steps, (self.num_envs, 1), device=self.device
            )
            self.action_delay_idx = torch.floor(action_delay_steps).squeeze(-1).long()
            self.action_delay_idx = torch.clamp(self.action_delay_idx, min=0, max=delay_max - 1)

        # --- 7. Initialize Command Buffers ---
        self.command_ranges = {
            "lin_vel_x": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False),
            "lin_vel_y": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False),
            "ang_vel_yaw": torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False),
        }
        # Copy values from config - Ensure self.cfg is fully initialized before this point
        self.command_ranges["lin_vel_x"][:] = torch.tensor(self.cfg.commands.ranges.lin_vel_x, device=self.device)
        self.command_ranges["lin_vel_y"][:] = torch.tensor(self.cfg.commands.ranges.lin_vel_y, device=self.device)
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(self.cfg.commands.ranges.ang_vel_yaw, device=self.device)

        if self.cfg.commands.heading_command:
             self.command_ranges["heading"] = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
             if hasattr(self.cfg.commands.ranges, 'heading'):
                  self.command_ranges["heading"][:] = torch.tensor(self.cfg.commands.ranges.heading, device=self.device)
             else: # Default range
                  self.command_ranges["heading"][:] = torch.tensor([-np.pi, np.pi], device=self.device)

        num_cmd_dims = self.cfg.commands.num_commands
        if self.cfg.commands.heading_command:
            num_cmd_dims += 1
        self.commands = torch.zeros(self.num_envs, num_cmd_dims, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        # TODO: Add heading scale if heading_command is true?

        # --- 8. Initialize DOF position offsets and PD gains ---
        self.raw_default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        # --- 9. Apply Domain Randomization (Gains, Torque Scale, Default Pos) --- Added back
        self.torques_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_motor_torque:
            torque_scale_min, torque_scale_max = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(torque_scale_min, torque_scale_max, self.torques_scale.shape, device=self.device)

        if self.cfg.domain_rand.randomize_Kp:
            p_gains_scale_min, p_gains_scale_max = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(p_gains_scale_min, p_gains_scale_max, self.p_gains.shape, device=self.device)

        if self.cfg.domain_rand.randomize_Kd:
            d_gains_scale_min, d_gains_scale_max = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(d_gains_scale_min, d_gains_scale_max, self.d_gains.shape, device=self.device)

        if self.cfg.domain_rand.randomize_default_dof_pos:
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )

        # IMU offset randomization (Added)
        if self.cfg.domain_rand.randomize_imu_offset:
            min_angle, max_angle = self.cfg.domain_rand.randomize_imu_offset_range
            min_angle_rad = np.radians(min_angle)
            max_angle_rad = np.radians(max_angle)
            pitch = torch_rand_float(min_angle_rad, max_angle_rad, (self.num_envs,), device=self.device)
            roll = torch_rand_float(min_angle_rad, max_angle_rad, (self.num_envs,), device=self.device)
            pitch_quat = torch.stack([torch.zeros_like(pitch), torch.sin(pitch / 2), torch.zeros_like(pitch), torch.cos(pitch / 2)], dim=-1)
            roll_quat = torch.stack([torch.sin(roll / 2), torch.zeros_like(roll), torch.zeros_like(roll), torch.cos(roll / 2)], dim=-1)
            self.random_imu_offset = quat_mul(pitch_quat, roll_quat)
        else:
            self.random_imu_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # --- 10. Initialize Foot State Buffers ---
        if hasattr(self, 'feet_indices'):
            num_feet = len(self.feet_indices)
            self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
            self.last_foot_positions = torch.zeros_like(self.foot_positions)
            self.foot_heights = torch.zeros_like(self.foot_positions[:, :, 0]) # Height is scalar per foot
            self.foot_velocities = torch.zeros_like(self.foot_positions)
            self.foot_velocities_f = torch.zeros_like(self.foot_positions)
            self.foot_relative_velocities = torch.zeros_like(self.foot_velocities)
            self.feet_air_time = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False) # Added
            self.last_contacts = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device, requires_grad=False) # Added
        else:
            print("Warning: feet_indices not found during foot buffer init. Initializing foot buffers as empty/zero.")
            num_feet = 0
            # Initialize empty/zero tensors to prevent errors later if accessed
            self.foot_positions = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.last_foot_positions = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.foot_heights = torch.empty((self.num_envs, 0), device=self.device)
            self.foot_velocities = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.foot_velocities_f = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.foot_relative_velocities = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.feet_air_time = torch.empty((self.num_envs, 0), device=self.device)
            self.last_contacts = torch.empty((self.num_envs, 0), dtype=torch.bool, device=self.device)

        # --- 11. Initialize Ball-Specific Buffers ---
        if self.num_balls > 0:
            self.ball_positions = self.root_states[:, 1:, :3]
            self.ball_linvels = self.root_states[:, 1:, 7:10]
        else:
            self.ball_positions = torch.empty((self.num_envs, 0, 3), device=self.device)
            self.ball_linvels = torch.empty((self.num_envs, 0, 3), device=self.device)

        self.ball_radius = torch_rand_float(self.cfg.ball.radius_range[0], self.cfg.ball.radius_range[1], (self.num_envs, self.num_balls), device=self.device)
        self.ball_mass = torch_rand_float(self.cfg.ball.mass_range[0], self.cfg.ball.mass_range[1], (self.num_envs, self.num_balls), device=self.device)

        # --- 12. Initialize Observation Buffers ---
        # Calculate obs size based on components (adjust as needed)
        proprioceptive_obs_size = 30 # Example, verify actual size
        ball_obs_size_per_ball = 7  # Example, verify actual size
        self.num_observations = proprioceptive_obs_size + self.num_balls * ball_obs_size_per_ball

        self.obs_buf = torch.zeros(self.num_envs, self.num_observations, device=self.device, dtype=torch.float)
        self.obs_history_length = getattr(self.cfg.env, 'obs_history_length', 1)
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_observations * self.obs_history_length, device=self.device, dtype=torch.float)

        # --- 13. Initialize Force/Wrench/External Force Buffers ---
        self.wrenches_on_robot = torch.zeros(self.num_envs, 6, device=self.device)
        self.wrench_buffer = torch.zeros(self.num_envs, 50, 6, device=self.device) # History buffer
        self.force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim) # Note: Might need refresh?
        self.rigid_body_external_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # Added
        self.rigid_body_external_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # Added

        # --- 14. Initialize Noise Scaling ---
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # Assumes this helper is available/copied

        # --- 15. Initialize Gait/Clock Buffers ---
        self.gaits = torch.zeros(self.num_envs, self.cfg.gait.num_gait_params, dtype=torch.float, device=self.device, requires_grad=False)
        if num_feet > 0:
            self.desired_contact_states = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
            self.doubletime_clock_inputs_sin = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
            self.halftime_clock_inputs_sin = torch.zeros(self.num_envs, num_feet, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            # Already handled initialization with num_feet=0
            pass
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs_sin = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs_cos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # --- 16. Initialize Step Counter Buffer ---
        self.envs_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        # --- 17. Initialize Height Measurement Buffers --- Added
        self.measured_heights = torch.zeros(self.num_envs, device=self.device) # Initialize as zero or None
        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            # Use the logic from BaseTask._init_height_points directly here
            if hasattr(self.cfg.terrain, 'measured_points_y') and hasattr(self.cfg.terrain, 'measured_points_x'):
                 y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
                 x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
                 grid_x, grid_y = torch.meshgrid(x, y, indexing='ij') # Ensure correct indexing
                 self.num_height_points = grid_x.numel()
                 self.height_points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
                 self.height_points[:, :, 0] = grid_x.flatten()
                 self.height_points[:, :, 1] = grid_y.flatten()
                 print(f"Initialized height points buffer with {self.num_height_points} points per env.")
            else:
                 print("Warning: cfg.terrain.measure_heights is True, but measured_points_x/y not defined in config!")
                 self.num_height_points = 0
                 self.height_points = torch.empty(self.num_envs, 0, 3, device=self.device)
        else:
            self.num_height_points = 0
            self.height_points = torch.empty(self.num_envs, 0, 3, device=self.device) # Ensure it exists even if not used

        # --- 18. Initialize extras dict ---
        self.extras = {}

        print("--------------------------------")
        print("BipedPFBallBalance: Buffer initialization complete.")
        print(f"  num_height_points: {getattr(self, 'num_height_points', 'Not set')}")
        print("--------------------------------")

    # Add the override method here
    def _resample_commands(self, env_ids):
        """ Resamples commanded forward, lateral velocity and yaw rate. Overrides parent to handle tensor command_ranges.

            Args:
                env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        # Sample lin_vel_x
        print("command_ranges in _resample_commands:", self.command_ranges)
        print("shape of command_ranges in _resample_commands:", self.command_ranges["lin_vel_x"].shape)
        lower_x = self.command_ranges["lin_vel_x"][env_ids, 0]
        upper_x = self.command_ranges["lin_vel_x"][env_ids, 1]
        self.commands[env_ids, 0] = lower_x + torch.rand(len(env_ids), device=self.device) * (upper_x - lower_x)
        
        # Sample lin_vel_y
        lower_y = self.command_ranges["lin_vel_y"][env_ids, 0]
        upper_y = self.command_ranges["lin_vel_y"][env_ids, 1]
        self.commands[env_ids, 1] = lower_y + torch.rand(len(env_ids), device=self.device) * (upper_y - lower_y)
        
        # Sample ang_vel_yaw
        lower_yaw = self.command_ranges["ang_vel_yaw"][env_ids, 0]
        upper_yaw = self.command_ranges["ang_vel_yaw"][env_ids, 1]
        self.commands[env_ids, 2] = lower_yaw + torch.rand(len(env_ids), device=self.device) * (upper_yaw - lower_yaw)

        # Sample heading if enabled
        if self.cfg.commands.heading_command:
            # Check if 'heading' key exists, crucial because _parse_cfg might not have run yet
            # depending on initialization order, or config might be missing it.
            if "heading" in self.command_ranges:
                lower_heading = self.command_ranges["heading"][env_ids, 0]
                upper_heading = self.command_ranges["heading"][env_ids, 1]
                self.commands[env_ids, 3] = lower_heading + torch.rand(len(env_ids), device=self.device) * (upper_heading - lower_heading)
            else:
                # Handle missing heading range if heading_command is true
                print("Warning: _resample_commands called with heading_command=True, but 'heading' not in command_ranges!")
                # Set a default heading or use current heading? For now, just set to 0.
                self.commands[env_ids, 3] = 0.0 

        # Apply curriculum filtering (copied from parent)
        # Consider moving this threshold (0.2) to config if necessary
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # --- Need to override _get_noise_scale_vec ---
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [OVERRIDES PARENT] Must be adapted for the child's observation structure.
        """
        # Determine child observation size again (or use self.num_observations)
        proprioceptive_obs_size = 30
        ball_obs_size_per_ball = 7
        num_total_obs = proprioceptive_obs_size + self.num_balls * ball_obs_size_per_ball

        noise_vec = torch.zeros(num_total_obs, device=self.device) # Use correct total size

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        self.add_noise = self.cfg.noise.add_noise

        # Noise for robot part (indices 0 to 29, based on BipedPF)
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12:18] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # Assuming next 12 are actions? Check BipedPF compute_group_observations
        # noise_vec[18:30] = 0.0 # previous actions?

        # Noise for ball part (indices 30 onwards) - Apply appropriate scales if needed
        start_idx = proprioceptive_obs_size
        for i in range(self.num_balls):
             # Example: Add noise to relative position and velocity if desired
             # pos_start = start_idx + i * ball_obs_size_per_ball
             # vel_start = pos_start + 3
             # noise_vec[pos_start : pos_start + 3] = noise_scales.ball_pos * noise_level # Need config for ball_pos noise
             # noise_vec[vel_start : vel_start + 3] = noise_scales.ball_vel * noise_level # Need config for ball_vel noise
             pass # Set ball noise scales as needed

        print(f"Child _get_noise_scale_vec: Created noise vector of size {noise_vec.shape[0]}")
        return noise_vec
    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments.
           Uses correct actor indexing for set_dof_state_tensor_indexed.
        """
        if len(env_ids) == 0:
            return

        # Calculate new DOF positions/velocities for the robot
        pos_range = getattr(self.cfg.init_state, 'dof_init_pos_range', [0.5, 1.5])
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids, :] * torch_rand_float(
            pos_range[0], pos_range[1], (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        # --- Calculate ROBOT indices --- 
        # This calculation assumes the global dof_state tensor is ordered like:
        # [env0_robot_dofs, env0_ball_dofs, env1_robot_dofs, env1_ball_dofs, ...]
        # If the ball has 0 DOFs, its entry might be empty or just affect indexing.
        # We need the indices corresponding to the *start* of each robot's DOF block.
        # The Isaac Gym documentation suggests indexed functions expect *actor* indices.

        # --- Calculate ROBOT indices --- 
        # Ensure the global mapping from env_id to robot actor index exists
        if not hasattr(self, 'all_robot_indices'):
             print("[DEBUG Reset DOFs] Calculating all_robot_indices...")
             # Assuming robot is actor 0 in each env
             # Indices are: 0, num_actors, 2*num_actors, ...
             self.all_robot_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device) * self.num_actors 

        # Select the global robot actor indices for the envs being reset
        robot_indices = self.all_robot_indices[env_ids].flatten() # Get the specific actor indices
        indices_to_use = robot_indices.to(dtype=torch.int32)
        print(f"[DEBUG Reset DOFs] Using indices (robot_indices): shape={indices_to_use.shape}, device={indices_to_use.device}, min/max: {indices_to_use.min()}/{indices_to_use.max()}")

        # --- DEBUG CHECKS ---
        if hasattr(self, 'dof_state'):
            print(f"[DEBUG Reset DOFs] dof_state shape: {self.dof_state.shape}, is_contig: {self.dof_state.is_contiguous()}")
            if torch.isnan(self.dof_state).any(): print("[DEBUG Reset DOFs] CRITICAL: NaNs found in ENTIRE dof_state before setting!")
            if torch.isinf(self.dof_state).any(): print("[DEBUG Reset DOFs] CRITICAL: Infs found in ENTIRE dof_state before setting!")
            # Check the specific slice we just modified
            if not torch.equal(self.dof_state.view(self.num_envs, self.num_dof, 2)[env_ids, :, 0], self.dof_pos[env_ids]):
                 print("[DEBUG Reset DOFs] CRITICAL: dof_state pos view MISMATCH with self.dof_pos after update!")
            if not torch.equal(self.dof_state.view(self.num_envs, self.num_dof, 2)[env_ids, :, 1], self.dof_vel[env_ids]):
                 print("[DEBUG Reset DOFs] CRITICAL: dof_state vel view MISMATCH with self.dof_vel after update!")
        else:
            print("[DEBUG Reset DOFs] CRITICAL: self.dof_state not found!"); return
        if torch.isnan(self.dof_pos[env_ids]).any() or torch.isnan(self.dof_vel[env_ids]).any():
             print("[DEBUG Reset DOFs] CRITICAL: NaNs found in dof_pos or dof_vel slices being set!")
        # Add Inf check
        elif torch.isinf(self.dof_pos[env_ids]).any() or torch.isinf(self.dof_vel[env_ids]).any():
            print("[DEBUG Reset DOFs] CRITICAL: Infs found in dof_pos or dof_vel slices being set!")
        else:
            print("[DEBUG Reset DOFs] No NaNs or Infs found in dof_pos/vel slices.")
        self.dof_state = self.dof_state.contiguous()
        print(f"[DEBUG Reset DOFs] dof_state enforced contiguous: {self.dof_state.is_contiguous()}")
        # --- END DEBUG CHECKS ---

        # Call with the chosen indices
        print(f"[DEBUG Reset DOFs] Calling set_dof_state_tensor_indexed with {indices_to_use.numel()} indices...")
        try:
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(indices_to_use), # Use chosen indices
                len(indices_to_use), # Use length of chosen indices
            )
            print("[DEBUG Reset DOFs] set_dof_state_tensor_indexed call completed.")
        except Exception as e:
            print(f"[DEBUG Reset DOFs] ERROR during set_dof_state_tensor_indexed: {e}")
            raise e

        print("[DEBUG Reset DOFs] Attempting to print env_ids AFTER set_dof_state...")
        print("shape of env_ids in _reset_dofs:", env_ids.shape)
        print("env_ids in _reset_dofs, after gym.set_dof_state_tensor_indexed:", env_ids)
        print("[DEBUG Reset DOFs] Printing env_ids successful.")

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Override of BaseTask._post_physics_step_callback to handle multi-actor state,
            include necessary resampling/stepping calls, and correct base_height calculation.
        """
        # --- Resampling Logic from BaseTask --- 
        # Determine env_ids for resampling based on time
        resample_env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        # Resample commands and gaits for these specific envs
        if hasattr(self, '_resample_commands'):
            self._resample_commands(resample_env_ids)
        if hasattr(self, '_resample_gaits'):
             self._resample_gaits(resample_env_ids)
        if hasattr(self, '_step_contact_targets'):
             self._step_contact_targets() # This usually doesn't need env_ids
        # --- End Resampling Logic --- 

        # --- Heading Command Adjustment (Conditional) ---
        # Note: BaseTask doesn't condition this on env_ids length, 
        #       but it might be safer if heading depends on non-resetting state.
        #       Using the original BaseTask logic here:
        if self.cfg.commands.heading_command:
            # Use robot's state (actor 0)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # Assumes command index 3 is heading command
            self.commands[:, 2] = 1.0 * wrap_to_pi(self.commands[:, 3] - heading)

        # --- Height Measurement --- 
        # Compute measured heights if enabled
        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()
        # Else: self.measured_heights should have been initialized to zero

        # --- Corrected base_height Calculation --- 
        # Direct subtraction between robot's Z position (actor 0) and MINIMUM measured ground height
        if hasattr(self, 'measured_heights'): # Ensure measured_heights was initialized
            # Use robot's z-position from root_states
            robot_z_position = self.root_states[:, 0, 2]
            # Take the minimum across sampled points
            min_measured_heights = self.measured_heights.min(dim=1)[0] 
            self.base_height = robot_z_position - min_measured_heights
        else:
            # Fallback or error handling if measured_heights isn't available
            print("Warning: measured_heights not found in _post_physics_step_callback. Setting base_height based on root_state only.")
            self.base_height = self.root_states[:, 0, 2] # Or set to zero/handle appropriately

    # Existing methods below...
    def _reward_base_height(self):
        """ Penalize base height away from target, using the correctly computed self.base_height.
            Overrides the parent method.
        """
        # Use the pre-computed base_height (calculated correctly in _post_physics_step_callback)
        # self.base_height has shape [num_envs]
        return torch.square(self.base_height - self.cfg.rewards.base_height_target)