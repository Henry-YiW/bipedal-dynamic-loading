from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.pointfoot_flat.pointfoot_flat_config import BipedCfgPF, BipedCfgPPOPF
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

class BipedBallBalanceCfg(BipedCfgPF):
    class env(BipedCfgPF.env):
        # Inherit all locomotion params and extend for balls
        num_envs = 4096
        num_balls = 0  # Default number of balls per environment (can be overridden)
        # Dynamically calculate total observations based on num_balls in this class
        num_robot_observations = 30
        num_observations = num_robot_observations + (7 * num_balls)
        # Critic observations also depend on the dynamic num_observations
        # Note: Base class might calculate this differently, ensure consistency if needed.
        num_critic_observations = 3 + num_observations

    class asset: # Define the class directly, no inheritance here
        # Path to the PROCESSED PF_TRON1A robot URDF with fixed basket
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/PF_TRON1A/urdf/robot_with_basket_processed.urdf'
        name = "pointfoot_tron1a_with_basket"
        foot_name = "foot" # Check if this correctly finds foot_L_Link/foot_R_Link
        foot_radius = 0.03  # Approximate radius of the foot contact point (meters)
        penalize_contacts_on = ["hip_L_Link", "knee_L_Link", "hip_R_Link", "knee_R_Link", "contact_plate", "back_box"]
        terminate_after_contacts_on = ["base_Link", "abad_L_Link", "abad_R_Link"]
        # Explicitly list ALL necessary asset parameters here, as they are NOT inherited
        # Values below are typical defaults from BaseConfig or BipedCfgPF - adjust if needed
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0 # 0 to enable, 1 to disable
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        # You might need these too if BipedCfgPF defines them directly
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class ball:
        # Ball physical properties
        radius_range = [0.1, 0.15]  # Min/max ball radius (m)
        mass_range = [0.5, 1.5]     # Min/max ball mass (kg)
        spawn_radius = 0.5          # Radius for spawning balls around robot
        spawn_pattern = "circle"    # "circle", "random", or "fixed"
        max_distance = 1.0          # Terminate if balls exceed this distance (m)
        
        # Reward weights
        pos_weight = 2.0            # Weight for position error reward
        vel_weight = 0.1            # Weight for velocity penalty
        wrench_smoothness_sigma = 5.0  # Sigma for wrench smoothness reward

    class rewards(BipedCfgPF.rewards):
        class scales(BipedCfgPF.rewards.scales):
            # Preserve all locomotion rewards and add ball-specific ones
            ball_balance = 1.0       # Weight for ball balancing reward
            wrench_smoothness = 0.5  # Weight for force smoothness

    class terrain(BipedCfgPF.terrain):
        mesh_type = "plane"  # Default to plane for ball balancing
        curriculum = False   # Disable terrain curriculum for this task

    class commands(BipedCfgPF.commands):
        # Reduce command ranges for more precise control
        class ranges(BipedCfgPF.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-1.0, 1.0]

    class init_state(BipedCfgPF.init_state):
        # Start with robot standing still
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

    class control(BipedCfgPF.control):
        # Tighter control for balancing
        stiffness = {
            "abad_L_Joint": 50,  # Increased stiffness for better stability
            "hip_L_Joint": 50,
            "knee_L_Joint": 50,
            "abad_R_Joint": 50,
            "hip_R_Joint": 50,
            "knee_R_Joint": 50,
            "foot_L_Joint": 0.0,
            "foot_R_Joint": 0.0
        }
        damping = {
            "abad_L_Joint": 3.0,  # Increased damping
            "hip_L_Joint": 3.0,
            "knee_L_Joint": 3.0,
            "abad_R_Joint": 3.0,
            "hip_R_Joint": 3.0,
            "knee_R_Joint": 3.0,
            "foot_L_Joint": 0.0,
            "foot_R_Joint": 0.0
        }

class BipedBallBalanceCfgPPO(BipedCfgPPOPF):
    class runner(BipedCfgPPOPF.runner):
        # Longer training for complex task
        run_name = "pointfoot_ball_balance"
        num_steps_per_env = 24  # per iteration
        max_iterations = 7000  # number of policy updates

    # Override MLP_Encoder to set correct input dimension
    class MLP_Encoder(BipedCfgPPOPF.MLP_Encoder):
        # Dynamically calculate input dimension using the corresponding env config
        num_input_dim = BipedBallBalanceCfg.env.num_observations * BipedBallBalanceCfg.env.obs_history_length
        # Keep other params like hidden_dims, activation etc. inherited
        # or specify them if needed:
        # hidden_dims = [256, 128]
        # activation = "elu"
        # ...

    # Policy and Algorithm classes are inherited directly unless changes are needed.
    class policy(BipedCfgPPOPF.policy):
        # Explicitly define actor network dimensions
        # The ActorCritic class likely uses num_actor_obs passed during instantiation,
        # but we define the hidden dims here for clarity and potential override.
        # We assume the actual input size passed will be 36 (proprio + cmd + encoder_out)
        # Note: This might not directly change the Linear layer's in_features if
        # ActorCritic calculates it differently. We might need to adjust how
        # num_actor_obs is determined when ActorCritic is created in train.py.
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        # Set init_noise_std if needed
        init_noise_std = 1.0
        pass # Add other policy overrides if necessary

    # class algorithm(BipedCfgPPOPF.algorithm):
    #     pass # Or override specifics