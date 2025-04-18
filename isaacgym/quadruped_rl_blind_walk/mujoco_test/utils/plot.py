import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("/home/ubuntu/quadruped_load_adaption/observation_student_adapt_ball_left.txt")
data_adapt = np.loadtxt("/home/ubuntu/quadruped_load_adaption/observation_student_adapt_backward.txt")
time = data[:, 0]
command_x = data[:, 7]
x_velocity = data[:, 1]
y_velocity = data[:, 2]
x_angular_velocity = data[:, 4]
y_angular_velocity = data[:, 5]
z_velocity = data[:, 6]
z_height = data[:, 8]
x_velocity_adapt = data_adapt[:, 1]
y_velocity_adapt = data_adapt[:, 2]
x_angular_velocity_adapt = data_adapt[:, 4]
y_angular_velocity_adapt = data_adapt[:, 5]
z_velocity_adapt = data_adapt[:, 6]
z_height_adapt = data_adapt[:, 8]
# Plot x_velocity, y_velocity, roll, pitch, and z_velocity vs. time
plt.figure(figsize=(10, 10))
plt.subplot(4, 1, 1)
plt.plot(time, x_velocity, label="base_lin_vel_x")
plt.plot(time, np.full_like(time, 0.0), label="command_x")
plt.xlabel("Time")
plt.ylabel("Base Linear Velocity (x)")
plt.title("Base Linear Velocity (x) vs. Time")
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(time, y_velocity, label="base_lin_vel_y")
plt.plot(time, command_x, label="command_y")
plt.xlabel("Time")
plt.ylabel("Base Linear Velocity (y)")
plt.title("Base Linear Velocity (y) vs. Time")
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(time, x_angular_velocity, label="base_ang_vel_x")
plt.plot(time, y_angular_velocity, label="base_ang_vel_y")
plt.plot(time, z_velocity, label="base_ang_vel_z")
plt.plot(time, np.full_like(time, 0), label="command_w")
plt.xlabel("Time")
plt.ylabel("Base Angular Velocity (z)")
plt.title("Base Angular Velocity (z) vs. Time")
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(time, z_height, label="z_height")
plt.xlabel("Time")
plt.ylabel("z_height")
plt.title("z_height vs. Time")
plt.legend()
plt.tight_layout()
plt.show()
# plt.figure(figsize=(15, 15))
# plt.subplot(4, 1, 1)
# plt.plot(time, x_velocity, label='base_lin_vel_x')
# plt.plot(time, x_velocity_adapt, label='base_lin_vel_x_adapt')
# plt.plot(time, command_x, label='command_x')
# plt.xlabel('Time')
# plt.ylabel('Base Linear Velocity (x)')
# plt.title('Base Linear Velocity (x) vs. Time')
# plt.legend()
# plt.subplot(4, 1, 2)
# plt.plot(time, y_velocity, label='base_lin_vel_y')
# plt.plot(time, y_velocity_adapt, label='base_lin_vel_y_adapt')
# plt.plot(time, np.full_like(time, -0.2), label='command_y')
# plt.xlabel('Time')
# plt.ylabel('Base Linear Velocity (y)')
# plt.title('Base Linear Velocity (y) vs. Time')
# plt.legend()
# plt.subplot(4, 1, 3)
# plt.plot(time, x_angular_velocity, label='base_ang_vel_x', linestyle='--')
# plt.plot(time, x_angular_velocity_adapt, label='base_ang_vel_x_adapt')
# plt.plot(time, y_angular_velocity, label='base_ang_vel_y', linestyle='--')
# plt.plot(time, y_angular_velocity_adapt, label='base_ang_vel_y_adapt')
# plt.plot(time, z_velocity, label='base_ang_vel_z', linestyle='--')
# plt.plot(time, z_velocity_adapt, label='base_ang_vel_z_adapt')
# plt.plot(time, np.full_like(time, 0), label='command_w')
# plt.xlabel('Time')
# plt.ylabel('Base Angular Velocity')
# plt.title('Base Angular Velocity vs. Time')
# plt.legend()
# plt.subplot(4, 1, 4)
# plt.plot(time, z_height, label='z_height')
# plt.plot(time, z_height_adapt, label='z_height_adapt')
# plt.xlabel('Time')
# plt.ylabel('z_height')
# plt.title('z_height vs. Time')
# plt.legend()
# plt.tight_layout()
# plt.show()
