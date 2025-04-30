from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from ..mixed_terrain.pointfoot_rough_config import PointFootRoughCfg, PointFootRoughCfgPPO

class PointFootWithLoadCfg(PointFootRoughCfg):
    class env(PointFootRoughCfg.env):
        num_envs = 512
        num_observations = 48
        num_privileged_obs = 48
        num_actions = 12
        episode_length_s = 20

    class terrain(PointFootRoughCfg.terrain):
        mesh_type = 'plane'
        curriculum = False

    class init_state(PointFootRoughCfg.init_state):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

    class control(PointFootRoughCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}    # [N*m*s/rad]
        decimation = 4

    class load:
        mass = 1.0  # [kg]
        size = [0.1, 0.1, 0.1]  # [m]
        init_pos = [0.0, 0.0, 0.1]  # [m] relative to base
        friction = 0.5
        restitution = 0.0
        use_container = True
        container_size = [0.2, 0.2, 0.2]  # [m]
        container_pos = [0.0, 0.0, 0.2]  # [m] relative to base
        container_friction = 0.5
        container_restitution = 0.0

    class asset(PointFootRoughCfg.asset):
        import os
        os.environ["ROBOT_TYPE"] = "PF_TRON1A"  # Set the robot type
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pointfoot/PF_TRON1A/urdf/robot.urdf'

    class sim(PointFootRoughCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 4
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**20
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class rewards(PointFootRoughCfg.rewards):
        class scales(PointFootRoughCfg.rewards.scales):
            # load_movement_penalty = 0.1 # Removed, seems redundant with stability/position
            load_stability = 0.2
            load_position = 0.2
            container_stability = 0.2
            container_position = 0.2
            dof_pos_limits = 0.9
            base_height = 0.5
            feet_contact_forces = 100.
            
        # Load reward parameters
        load_stability_sigma = 0.1
        load_position_sigma = 0.1
            
        # Container reward parameters
        container_stability_sigma = 0.1
        container_position_sigma = 0.1

    class domain_rand(PointFootRoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [0.0, 1.0]
        randomize_base_com = True
        rand_com_vec = [0.1, 0.1, 0.1]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.

class PointFootWithLoadCfgPPO(PointFootRoughCfgPPO):
    class algorithm(PointFootRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(PointFootRoughCfgPPO.runner):
        run_name = 'pointfoot_with_load'
        experiment_name = 'pointfoot_with_load' 