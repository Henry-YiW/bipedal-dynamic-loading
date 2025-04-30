import os

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO # Use base PPO config

# Define the load parameters
class LoadParams:
    mass_range = [0.1, 2.0]  # kg, range for randomizing load mass
    friction_coefficient_range = [0.3, 0.9]  # range for randomizing load friction

class PointFootBallBalanceCfg(BaseConfig):
    class env:
        num_envs = 4096
        # PF_TRON1A: ang_vel(3)+grav(3)+cmd(3)+dof_pos(6)+dof_vel(6)+actions(6)=27
        num_propriceptive_obs = 27 
        # PF_TRON1A: base_obs(27)+friction(1)+load_obs(8)=36. No heights.
        num_privileged_obs = 36 
        num_actions = 6 # PF_TRON1A has 6 actuated joints
        num_actors = 2  # Robot + Ball
        num_obs_load = 8 # [rel_pos(3), rel_vel(3), mass(1), friction(1)]
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 20 # Uncommented to enable history buffer

        # Set measure_heights flags to False
        measure_heights_actor = False
        measure_heights_critic = False

    # Ball initial state definition
    class ball_init_state:
        pos = [0.5, 0.0, 0.15]  # x,y,z [m] relative to robot base center
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

    class terrain:
        mesh_type = 'plane'  # Use plane for simplicity first
        horizontal_scale = 0.1  # [m] - Not used for plane
        vertical_scale = 0.005  # [m] - Not used for plane
        border_size = 25  # [m] - Not used for plane
        curriculum = False # No curriculum for plane
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights_actor = False
        measure_heights_critic = False
        # No need for measured points if not measuring heights

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3  # lin_vel_x, lin_vel_y, ang_vel_yaw (heading is internal)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s] # Wider range than rough config
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14] # For heading command calculation

    class init_state:
        pos = [0.0, 0.0, 0.40]  # x,y,z [m] - Adjusted height for PF_TRON1A?
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # Default joint angles for PF_TRON1A (6 joints)
        default_joint_angles = {
            'abad_L_Joint': 0.0,
            'hip_L_Joint': 0.0, 
            'knee_L_Joint': 0.0,
            'abad_R_Joint': 0.0,
            'hip_R_Joint': 0.0, 
            'knee_R_Joint': 0.0,
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters for PF_TRON1A (6 joints)
        stiffness = {
            'abad_L_Joint': 40.0, 
            'hip_L_Joint': 40.0, 
            'knee_L_Joint': 40.0,
            'abad_R_Joint': 40.0, 
            'hip_R_Joint': 40.0, 
            'knee_R_Joint': 40.0,
        }  # [N*m/rad] - Uniform stiffness for now
        damping = {
            'abad_L_Joint': 1.0, 
            'hip_L_Joint': 1.0, 
            'knee_L_Joint': 1.0,
            'abad_R_Joint': 1.0, 
            'hip_R_Joint': 1.0, 
            'knee_R_Joint': 1.0,
        }    # [N*m*s/rad] - Uniform damping for now
        action_scale = 0.5 # Adjust as needed
        decimation = 4

    class asset:
        # Path to the PROCESSED PF_TRON1A robot URDF with fixed basket
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pointfoot/PF_TRON1A/urdf/robot_with_basket_processed.urdf' # Use the PROCESSED TRON1A basket URDF
        name = "pointfoot_tron1a_with_basket" # Name remains the same
        foot_name = "foot" # Assuming code finds foot_L_Link/foot_R_Link
        # Contact penalization/termination based on PF_TRON1A link names
        penalize_contacts_on = ["hip_L_Link", "knee_L_Link", "hip_R_Link", "knee_R_Link"]
        terminate_after_contacts_on = ["base_Link", "abad_L_Link", "abad_R_Link"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False # Usually false for STL

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25] # Base range
        randomize_base_mass = False # Keep false for now
        added_mass_range = [-0.1, 0.1]
        randomize_base_com = False # Keep false for now
        rand_com_vec = [0.01, 0.01, 0.01]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    # Add LoadParams class for ball randomization parameters
    load_params = LoadParams()

    class rewards:
        soft_dof_pos_limit = 0.95 # From rough config
        soft_dof_vel_limit = 0.9 # From rough config
        soft_torque_limit = 0.8 # From rough config
        base_height_target = 0.40 # Match adjusted init_state pos z
        max_contact_force = 100. # Lower than rough config? Review needed.
        only_positive_rewards = False
        min_feet_air_time = 0.25 # From rough config
        max_feet_air_time = 0.65 # From rough config
        tracking_sigma = 0.25

        class scales: # Keep scales as before, but ensure reward funcs exist
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0 # From rough config

            # Regularization rewards
            torques = -0.00002 # Similar to rough
            dof_vel = -0.0 # Keep zero for now
            dof_acc = -2.5e-7 # Similar to rough
            action_rate = -0.01

            # Contact/Gait rewards
            feet_air_time = 1.0 # Keep simple version for now
            collision = -1.0 # Penalize base/thigh/calf collisions
            # feet_stumble = -0.0 # Keep off for now
            stand_still = -0.0 # Keep off for now
            # feet_contact_forces = -0.01 # Optional penalty from rough

            # Limit rewards
            dof_pos_limits = -1.0 # Penalize hitting joint limits
            torque_limits = -0.1 # From rough config

            # Termination reward
            termination = -0.0 # Typically zero

            # Ball specific rewards
            ball_dist = 2.0 # Reward for minimizing distance to ball
            ball_lin_vel = -0.5 # Penalize ball linear velocity

            # Removed/Zeroed from rough config:
            # feet_distance, no_fly, unbalance_feet_air_time, unbalance_feet_height, survival

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0 # Include scale even if not measuring

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1 # Include scale even if not measuring

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6] # Standard camera pos
        lookat = [0, 0, 1.] # Look at origin

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81] # [m/s^2]
        up_axis = 1 # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1 # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01 # [m]
            rest_offset = 0.0 # [m]
            bounce_threshold_velocity = 0.5 # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


# Define the PPO configuration by inheriting from the base PPO config
class PointFootBallBalanceCfgPPO(LeggedRobotCfgPPO): # Inherit from base PPO
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        # Adjust other PPO parameters as needed
        # e.g., learning_rate = 1.e-3, num_learning_epochs = 5, num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'pointfoot_ball_balance' # Set experiment name
        # Adjust runner parameters as needed
        max_iterations = 1500 # Standard max iterations
        save_interval = 50 # Standard save interval

# Merge LoadParams into the environment config (no need to merge into PPO config unless runner uses it directly)
PointFootBallBalanceCfg.load_params = LoadParams() 