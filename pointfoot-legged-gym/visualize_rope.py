import os
import time
from isaacgym import gymapi

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# Use PhysX backend
# sim_params.physics_engine = gymapi.SIM_PHYSX
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.max_depenetration_velocity = 1.0

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Set camera position and look-at
# cam_pos = gymapi.Vec3(0.25, -0.8, 1.05)
# cam_target = gymapi.Vec3(-0.05, -0.2, 0)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)  # the rope's world position
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cam_pos = gymapi.Vec3(0.0, 0.0, 1.5)        # camera above the Z shape
# cam_target = gymapi.Vec3(0.0, 0.0, 0.2)      # look at the rope's base height
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)



# Create ground
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Set asset root directory
asset_root = os.path.abspath("resources")  # Change as needed
hand_file = "robots/PF_TRON1A/urdf/robot_with_basket_processed.urdf"
# rope_file = "mjcf/open_ai_assets/hand/rope.xml"

# Asset loading options
asset_opts = gymapi.AssetOptions()
asset_opts.fix_base_link = False
asset_opts.disable_gravity = False
asset_opts.armature = 0.01
asset_opts.thickness = 0.001
asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE

# Load assets
hand_asset = gym.load_asset(sim, asset_root, hand_file, asset_opts)
# rope_asset = gym.load_asset(sim, asset_root, rope_file, asset_opts)

# Create environment
env_spacing = 1.5
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Create actors
hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
hand_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
gym.create_actor(env, hand_asset, hand_pose, "shadow_hand", 0, 1)

rope_pose = gymapi.Transform()
rope_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
# gym.create_actor(env, rope_asset, rope_pose, "rigid_rope", 0, 2)

# Run simulation loop
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    time.sleep(1.0 / 60.0)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
