from legged_gym.envs.base.base_config import BaseConfig

class PointFootBallCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 45 + 3 + 187 + 63 + 12 + 12 + 12 
        num_actions = 6  # For bipedal robot (instead of 12 for quadruped)
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
        num_actors = 2  # 1 robot and 1 ball
        obs_history_length = 15
        num_obs_load = 8

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.3, 0.3, 0.1, 0.1, 0.2]
        slope_treshold = 0.6

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.2, 0.2]  # More restricted than quadruped
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.62]  # Adjusted height for biped
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {  # Biped joints
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class ball_init_state:
        pos = [0, 0, 1.3]
        rot = [0, 0, 0, 1]
        lin_vel = [0, 0, 0]
        ang_vel = [0, 0, 0]

    class control:
        control_type = 'P'
        stiffness = {  # Adjusted for biped
            "abad_L_Joint": 40.0,
            "hip_L_Joint": 40.0,
            "knee_L_Joint": 40.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 40.0,
            "hip_R_Joint": 40.0,
            "knee_R_Joint": 40.0,
            "foot_R_Joint": 0.0,
        }
        damping = {  # Adjusted for biped
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,
            "foot_R_Joint": 0.0,
        }
        action_scale = 0.5
        decimation = 4

    class asset:
        file = ""
        name = "pointfoot_robot"
        foot_name = "foot"
        penalize_contacts_on = ["abad", "base"]
        terminate_after_contacts_on = ["base", "abad", "hip", "knee"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_base_inertia = True
        added_inertia_range_xx = [-0.005,0.005]
        added_inertia_range_xy = [-0.00002,0.00002]
        added_inertia_range_xz = [-0.0002,0.0002]
        added_inertia_range_yy = [-0.02,0.02]
        added_inertia_range_zz = [-0.02,0.02]
        
        randomize_leg_mass = True
        added_leg_mass_range = [-0.05,0.05]
        factor_leg_mass_range = [0.85,1.15]
        
        randomize_leg_com = True
        added_leg_com_range = [-0.015, 0.015]
        
        randomize_friction = True
        friction_range = [0.05, 1.25]
        
        randomize_base_mass = True
        added_mass_range = [-1., 3.0]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 2.

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]

        randomize_action_delay = True
        delay_ms_range = [0, 10]
        
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]

    class load_params:
        mass_range = [0.1, 8]
        friction_coefficient_range = [0.001, 0.2]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 2.
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            power = -2e-5
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -1.2
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            action_smoothness = -0.001
            stand_still = -0.
            dof_pos_limits = -2.0

            # ball related
            ball_lin_vel = 2
            # ball_ang_vel = -1.
            # ball_dist = 0.05

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.62
        max_contact_force = 100.

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 10.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            adaptive_noise = 0.1

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class PointFootBallCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        experiment_name = 'pointfoot_ball'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None 