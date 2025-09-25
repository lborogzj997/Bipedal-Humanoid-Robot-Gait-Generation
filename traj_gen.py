import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# ----------------------
# Parameters
# ----------------------
t_start = 0
t_end = 1
dt = 0.005
A_x = 0.2
A_z = 0.3
A_y = 0.0  # Y axis inward swing amplitude
A_offset = 0.0  # Z offset amplitude
omega = np.pi
n_loops = 200
max_pitch_deg = 20
A_y_base = 0.15  # Base link Y-axis movement amplitude
A_z_base = 0
A_z_ratio = 1
A_roll_base_deg = 5  # Base link roll amplitude in degrees
A_roll_base = np.radians(A_roll_base_deg)  # Convert to radians

# Parameter to control swing duration
swing_ratio = 0.9
swing_time = swing_ratio
t_swing_start = 1.0 - swing_ratio
t_swing_end = 1.0

# Initial positions
x0_right = 0.06
y0_right = -0.17
z0_right = -0.95
x0_left = x0_right
y0_left = -y0_right
z0_left = z0_right

# times
times1 = np.linspace(t_start, t_end, int((t_end - t_start)/dt) + 1)
times2 = np.linspace(1, 2, int((2 - 1)/dt) + 1)
times3 = np.linspace(2, 3, int((3 - 2)/dt) + 1)
times4 = np.linspace(3, 4, int((4 - 3)/dt) + 1)

mask_r1 = (times1 >= t_swing_start) & (times1 <= t_swing_end)
mask_r3 = ((times3 - 2) >= t_swing_start) & ((times3 - 2) <= t_swing_end)
mask_l2 = ((times2 - 1) >= t_swing_start) & ((times2 - 1) <= t_swing_end)
mask_l4 = ((times4 - 3) >= t_swing_start) & ((times4 - 3) <= t_swing_end)

# right x
# 1
x_right_1 = np.zeros_like(times1)
mask_support_r1 = times1 < t_swing_start
x_right_1[mask_support_r1] = x0_right - A_x * (times1[mask_support_r1])
mask_swing_r1 = (times1 >= t_swing_start) & (times1 <= t_swing_end)
x_start_r1 = x_right_1[mask_support_r1][-1]
t_swing_local_r1 = (times1[mask_swing_r1] - t_swing_start) / swing_time
x_right_1[mask_swing_r1] = x_start_r1 + (x0_right + A_x - x_start_r1) * (1 - np.cos(omega * t_swing_local_r1)) / 2
# 2
x_right_2 = x_right_1[-1] - 2 * A_x * ((times2 - 1) / 1)
# 3
x_right_3 = np.zeros_like(times3)
mask_support_r3 = (times3 - 2) < t_swing_start
x_right_3[mask_support_r3] = x_right_2[-1] - 2 * A_x * (times3[mask_support_r3] - 2)
mask_swing_r3 = ((times3 - 2) >= t_swing_start) & ((times3 - 2) <= t_swing_end)
x_start_r3 = x_right_3[mask_support_r3][-1]
t_swing_local_r3 = (times3[mask_swing_r3] - 2 - t_swing_start) / swing_time
x_right_3[mask_swing_r3] = x_start_r3 + (x0_right + A_x - x_start_r3) * (1 - np.cos(omega * t_swing_local_r3)) / 2
# 4
x_right_4 = x_right_3[-1] - 2 * A_x * ((times4 - 3) / 1)


# left x
# 1
x0_left_1 = x0_left - A_x * (times1 / (t_end - t_start))
# 2
x0_left_2 = np.zeros_like(times2)
mask_support_l2 = (times2 - 1) < t_swing_start
x0_left_2[mask_support_l2] = x0_left_1[-1] - 2 * A_x * (times2[mask_support_l2]-1)
mask_swing_l2 = ((times2 - 1) >= t_swing_start) & ((times2 - 1) <= t_swing_end)
x_start_l2 = x0_left_2[mask_support_l2][-1]
t_swing_local_l2 = (times2[mask_swing_l2] - 1 - t_swing_start) / swing_time
x0_left_2[mask_swing_l2] = x_start_l2 + (x0_left + A_x - x_start_l2) * (1 - np.cos(omega * t_swing_local_l2)) / 2
# 3
x0_left_3 = x0_left_2[-1] - 2 * A_x * ((times3 - 2) / 1)
# 4
x0_left_4 = np.zeros_like(times4)
mask_support_l4 = (times4 - 3) < t_swing_start
x0_left_4[mask_support_l4] = x0_left_3[-1] - 2 * A_x * (times4[mask_support_l4] - 3)
mask_swing_l4 = ((times4 - 3) >= t_swing_start) & ((times4 - 3) <= t_swing_end)
x_start_l4 = x0_left_4[mask_support_l4][-1]
t_swing_local_l4 = (times4[mask_swing_l4] - 3 - t_swing_start) / swing_time
x0_left_4[mask_swing_l4] = x_start_l4 + (x0_left + A_x - x_start_l4) * (1 - np.cos(omega * t_swing_local_l4)) / 2

# right y
# 1
y_right_1 = np.full_like(times1, y0_right)
y_right_1[mask_r1] = y0_right + A_y * (np.sin(omega * ((times1[mask_r1] - t_swing_start) / swing_time) / 2 - omega/2) + 1)
# 2
y_right_2 = y_right_1[-1] * np.ones_like(times2)
# 3
y_right_3 = y_right_2[-1] * np.ones_like(times3)
y_right_3[mask_r3] = y0_right + A_y - A_y * np.sin(omega * ((times3[mask_r3] - 2 - t_swing_start) / swing_time))
# 4
y_right_4 = y_right_3[-1] * np.ones_like(times4)

#left y
# 1
y0_left_1 = y0_left * np.ones_like(times1)
# 2
y0_left_2 = np.full_like(times1, y0_left)
y0_left_2[mask_l2] = y0_left - A_y * (np.sin(omega * ((times2[mask_l2] - 1 - t_swing_start) / swing_time) / 2 - omega/2) + 1)
# 3
y0_left_3 = y0_left_2[-1] * np.ones_like(times3)
# 4
y0_left_4 = y0_left_3[-1] * np.ones_like(times3)
y0_left_4[mask_l4] = y0_left - A_y + A_y * np.sin(omega * ((times3[mask_l4] - 3 - t_swing_start) / swing_time))

# right z
# 1
z_right_1 = np.full_like(times1, z0_right)
z_right_1[mask_r1] = z0_right + A_z * np.sin(omega * (times1[mask_r1] - t_swing_start) / swing_time)
# 2
z_right_2 = z_right_1[-1] * np.ones_like(times2)
# 3
z_right_3 = np.full_like(times3, z_right_2[-1])
z_right_3[mask_r3] = z_right_2[-1] + A_z * np.sin(omega * ((times3[mask_r3] - 2 - t_swing_start) / swing_time))
# 4
z_right_4 = z_right_3[-1] * np.ones_like(times4)

# left z
# 1
z0_left_1 = z0_left * np.ones_like(times1)
# 2
z0_left_2 = np.full_like(times2, z0_left)
z0_left_2[mask_l2] = z0_left + A_z * np.sin(omega * ((times2[mask_l2] - 1 - t_swing_start) / swing_time))
# 3
z0_left_3 = z0_left_2[-1] * np.ones_like(times3)
# 4
z0_left_4 = np.full_like(times4, z0_left_3[-1])
mask_l4 = ((times4 - 3) >= t_swing_start) & ((times4 - 3) <= t_swing_end)
z0_left_4[mask_l4] = z0_left_3[-1] + A_z * np.sin(omega * ((times4[mask_l4] - 3 - t_swing_start) / swing_time))

# foot pitch right
# 1
pitch_right_1 = np.zeros_like(times1)
pitch_mask_r1 = (times1 >= t_swing_start) & (times1 <= t_swing_end)
pitch_right_1[pitch_mask_r1] = max_pitch_deg * np.sin(omega * (times1[pitch_mask_r1] - t_swing_start) / swing_time)
# 2
pitch_right_2 = np.zeros_like(times2)
# 3
pitch_right_3 = np.zeros_like(times3)
pitch_mask_r3 = ((times3 - 2) >= t_swing_start) & ((times3 - 2) <= t_swing_end)
pitch_right_3[pitch_mask_r3] = max_pitch_deg * np.sin(omega * ((times3[pitch_mask_r3] - 2 - t_swing_start) / swing_time))
# 4
pitch_right_4 = np.zeros_like(times4)

# foot pitch left
# 1
pitch_left_1 = np.zeros_like(times1)
# 2
pitch_left_2 = np.zeros_like(times2)
pitch_mask_l2 = ((times2 - 1) >= t_swing_start) & ((times2 - 1) <= t_swing_end)
pitch_left_2[pitch_mask_l2] = max_pitch_deg * np.sin(omega * ((times2[pitch_mask_l2] - 1 - t_swing_start) / swing_time))
# 3
pitch_left_3 = np.zeros_like(times3)
# 4
pitch_left_4 = np.zeros_like(times4)
pitch_mask_l4 = ((times4 - 3) >= t_swing_start) & ((times4 - 3) <= t_swing_end)
pitch_left_4[pitch_mask_l4] = max_pitch_deg * np.sin(omega * ((times4[pitch_mask_l4] - 3 - t_swing_start) / swing_time))

# ----------------------
# Combine Phases
# ----------------------
times = np.concatenate((times1, times2, times3, times4))
x_right = np.concatenate((x_right_1, x_right_2, x_right_3, x_right_4))
y_right = np.concatenate((y_right_1, y_right_2, y_right_3, y_right_4))
z_right = np.concatenate((z_right_1, z_right_2, z_right_3, z_right_4))
pitch_right = np.concatenate((pitch_right_1, pitch_right_2, pitch_right_3, pitch_right_4))

x0_left_pos = np.concatenate((x0_left_1, x0_left_2, x0_left_3, x0_left_4))
y0_left_pos = np.concatenate((y0_left_1, y0_left_2, y0_left_3, y0_left_4))
z0_left_pos = np.concatenate((z0_left_1, z0_left_2, z0_left_3, z0_left_4))
pitch_left = np.concatenate((pitch_left_1, pitch_left_2, pitch_left_3, pitch_left_4))

# ----------------------
# Looping Phase (3–5s repeated)
# ----------------------
loop_times = np.concatenate((times3, times4)) - 2.0
loop_x_right = np.concatenate((x_right_3, x_right_4))
loop_y_right = np.concatenate((y_right_3, y_right_4))
loop_z_right = np.concatenate((z_right_3, z_right_4))
loop_pitch_right = np.concatenate((pitch_right_3, pitch_right_4))

loop_x0_left = np.concatenate((x0_left_3, x0_left_4))
loop_y0_left = np.concatenate((y0_left_3, y0_left_4))
loop_z0_left = np.concatenate((z0_left_3, z0_left_4))
loop_pitch_left = np.concatenate((pitch_left_3, pitch_left_4))

for i in range(n_loops):
    t_offset = 4 + i * 2
    times = np.concatenate((times, loop_times + t_offset))
    x_right = np.concatenate((x_right, loop_x_right))
    y_right = np.concatenate((y_right, loop_y_right))
    z_right = np.concatenate((z_right, loop_z_right))
    pitch_right = np.concatenate((pitch_right, loop_pitch_right))
    x0_left_pos = np.concatenate((x0_left_pos, loop_x0_left))
    y0_left_pos = np.concatenate((y0_left_pos, loop_y0_left))
    z0_left_pos = np.concatenate((z0_left_pos, loop_z0_left))
    pitch_left = np.concatenate((pitch_left, loop_pitch_left))

# ----------------------
# Z offset
# ----------------------
z_offset = np.zeros_like(times)
mask_0_1 = times <= 1
z_offset[mask_0_1] = A_offset * (1 - np.cos(np.pi * times[mask_0_1]))
mask_after_1 = times > 1
z_offset[mask_after_1] = A_offset * (1 - np.cos(2 * np.pi * (times[mask_after_1] - 1) + np.pi))

z_right += z_offset
z0_left_pos += z_offset

A_y_ratio = 0.9
base_y_swing_time = A_y_ratio
t_base_y_swing_start = 0
t_base_y_swing_end = base_y_swing_time
# 1
base_link_y1 = np.full_like(times1, A_y_base)
base_link_y_mask1 = (times1 >= t_base_y_swing_start) & (times1 <= t_base_y_swing_end)
base_link_y1[base_link_y_mask1] = A_y_base * np.sign(np.sin(0.5 * np.pi * times1[base_link_y_mask1] / base_y_swing_time)) * np.abs(np.sin(0.5 * np.pi * times1[base_link_y_mask1] / base_y_swing_time)) ** 0.7
# 2
base_link_y2 = np.full_like(times2, -A_y_base)
mask_y2 = ((times2 - 1.0) >= t_base_y_swing_start) & ((times2 - 1.0) <= t_base_y_swing_end)
base_link_y2[mask_y2] = A_y_base * np.sign(np.cos(np.pi * (times2[mask_y2] - 1) / base_y_swing_time)) * np.abs(np.cos(np.pi * (times2[mask_y2] - 1) / base_y_swing_time)) ** 0.7
# 3
base_link_y3 = np.full_like(times3, A_y_base)
mask_y3 = ((times3 - 2.0) >= t_base_y_swing_start) & ((times3 - 2.0) <= t_base_y_swing_end)
base_link_y3[mask_y3] =  -A_y_base * np.sign(np.cos(np.pi * (times3[mask_y3] - 2) / base_y_swing_time)) * np.abs(np.cos(np.pi * (times3[mask_y3] - 2) / base_y_swing_time)) ** 0.7
# 4
base_link_y4 = np.full_like(times4, -A_y_base)
mask_y4 = ((times4 - 3.0) >= t_base_y_swing_start) & ((times4 - 3.0) <= t_base_y_swing_end)
base_link_y4[mask_y4] = A_y_base * np.sign(np.cos(np.pi * (times4[mask_y4] - 3) / base_y_swing_time)) * np.abs(np.cos(np.pi * (times4[mask_y4] - 3) / base_y_swing_time)) ** 0.7

# Base link Y, Z, and Roll
base_link_y = np.concatenate((base_link_y1, base_link_y2, base_link_y3, base_link_y4))

# base_link_y = np.concatenate((
#     base_link_y1,
#     A_y_base * np.sign(np.cos(np.pi * (times2 - 1))) * np.abs(np.cos(np.pi * (times2 - 1))) ** 0.7,
#     -A_y_base * np.sign(np.cos(np.pi * (times3 - 2))) * np.abs(np.cos(np.pi * (times3 - 2))) ** 0.7,
#     A_y_base * np.sign(np.cos(np.pi * (times4 - 3))) * np.abs(np.cos(np.pi * (times4 - 3))) ** 0.7
# ))


base_link_z = np.full_like(times, -A_z_base)
start_offset = 1
for i, t in enumerate(times):
    if t >= start_offset:
        t_shifted = t - start_offset
        t_mod = t_shifted % 1.0  # Time within each 1s period, starting from offset
        if t_mod < A_z_ratio:
            # Half sine wave: 0 → -A_z_base → 0
            base_link_z[i] = -A_z_base+A_z_base * np.sin(np.pi * t_mod / A_z_ratio)
        else:
            base_link_z[i] = -A_z_base
    else:
        base_link_z[i] = -A_z_base

base_roll = np.concatenate((
    -A_roll_base * np.sin(0.5 * np.pi * times1),
    -A_roll_base * np.cos(np.pi * (times2 - 1)),
    A_roll_base * np.cos(np.pi * (times3 - 2)),
    -A_roll_base * np.cos(np.pi * (times4 - 3))
))

loop_base_y = np.concatenate((base_link_y[-len(times3)-len(times4):-len(times4)], base_link_y[-len(times4):]))
loop_base_z = np.concatenate((base_link_z[-len(times3)-len(times4):-len(times4)], base_link_z[-len(times4):]))
loop_roll = np.concatenate((base_roll[-len(times3)-len(times4):-len(times4)], base_roll[-len(times4):]))

for _ in range(n_loops):
    base_link_y = np.concatenate((base_link_y, loop_base_y))
    base_link_z = np.concatenate((base_link_z, loop_base_z))
    base_roll = np.concatenate((base_roll, loop_roll))

base_link_x = np.zeros_like(base_link_y)

# Write to CSV
with open('foot_trajectory.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    for i in range(len(times)):
        writer.writerow([
            times[i],
            x_right[i], y_right[i], z_right[i], pitch_right[i],
            x0_left_pos[i], y0_left_pos[i], z0_left_pos[i], pitch_left[i],
            base_link_x[i], base_link_y[i], base_link_z[i], base_roll[i]
        ])



# ----------------------
# Animation (X-Z View of both feet)
# ----------------------
trail_length = 30
alpha_decay = np.linspace(1.0, 0.1, trail_length)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-1.1, -0.8)
ax.set_aspect('equal')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Z Position (m)')
ax.set_title('Animated Foot Trajectories (X-Z View)')
ax.grid(True)

# Initialize trails for right foot and left foot
right_dots = [ax.plot([], [], 'o', color='blue', alpha=0)[0] for _ in range(trail_length)]
left_dots = [ax.plot([], [], 'o', color='red', alpha=0)[0] for _ in range(trail_length)]

def update(frame):
    # Right foot trail
    for i in range(trail_length):
        idx = frame - i
        if idx >= 0:
            right_dots[i].set_data([x_right[idx]], [z_right[idx]])
            right_dots[i].set_alpha(alpha_decay[i])
        else:
            right_dots[i].set_alpha(0)
    # Left foot trail
    for i in range(trail_length):
        idx = frame - i
        if idx >= 0:
            left_dots[i].set_data([x0_left_pos[idx]], [z0_left_pos[idx]])
            left_dots[i].set_alpha(alpha_decay[i])
        else:
            left_dots[i].set_alpha(0)
    return right_dots + left_dots

ani = animation.FuncAnimation(
    fig, update, frames=len(times), interval=10, blit=True, repeat=False
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Right Foot', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Left Foot', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements)

plt.show()