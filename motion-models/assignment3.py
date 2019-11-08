import numpy as np
import random
import matplotlib.pyplot as plt

def radians(angle):
    return angle/180.0 * np.pi

def plot(poses1, poses2, poses3, count):
    for i in range(len(poses1)):
        x = round(poses1[i][0], 2)
        y = round(poses1[i][1], 2)
        plt.plot(x, y, 'bo')
        # plt.text(x * (1 + 0.01), y * (1 + 0.01) , f"{count[i]/np.sum(count)}", fontsize=10)
        # plt.text(x * (1 + 0.01), y * (1 + 0.01) , f"{count[i]/np.sum(count)}", fontsize=10)

    if poses2:
        for i in range(len(poses2)):
            x = round(poses2[i][0], 2)
            y = round(poses2[i][1], 2)
            plt.plot(x, y, 'go')

    if poses3:
        for i in range(len(poses3)):
            x = round(poses3[i][0], 2)
            y = round(poses3[i][1], 2)
            plt.plot(x, y, 'ro')

    plt.plot(5, -3, 'yo')

    # plt.xlim((-5, 15))
    # plt.ylim((-16, 3))
    # plt.gca().invert_yaxis()
    plt.show()

def filter(poses):
    poses = np.delete(poses, np.s_[-1], 1)
    unique, counts = np.unique(poses, axis = 0, return_counts=True)
    print(unique, counts)
    return unique, counts

# Initial pose
start_pose = [5, -3, 45]

# Motion
motion = [-20, -30, 3]

# Errors
rot1_errors = [-10, 10]
rot2_errors = [-5, 5]
trans_errors = [-0.5, 0.5]

# First motion
poses1 = []

# Compute resulting pose
# due to errors
for rot1_error in rot1_errors:
    for rot2_error in rot2_errors:
        for trans_error in trans_errors:
            x = start_pose[0] + (motion[2] + trans_error) * np.cos(np.radians(start_pose[2] + (motion[0] + rot1_error)))
            y = start_pose[1] + (motion[2] + trans_error) * np.sin(np.radians(start_pose[2] + (motion[0] + rot1_error)))
            theta = start_pose[2] + (motion[0] + rot1_error) + (motion[1] + rot2_error)
            poses1.append([x, y, theta])

print(poses1)
unique, count = filter(poses1)
# plot(unique, count)
plot(poses1, None, None, count)

poses2 = []
for pose in poses1:
    for rot1_error in rot1_errors:
        for rot2_error in rot2_errors:
            for trans_error in trans_errors:
                x = pose[0] + (motion[2] + trans_error) * np.cos(np.radians(pose[2] + (motion[0] + rot1_error)))
                y = pose[1] + (motion[2] + trans_error) * np.sin(np.radians(pose[2] + (motion[0] + rot1_error)))
                theta = pose[2] + (motion[0] + rot1_error) + (motion[1] + rot2_error)
                poses2.append([x, y, theta])

unique, count = filter(poses2)
# plot(unique, count)
plot(poses1, poses2, None, count)

poses3 = []
for pose in poses2:
    for rot1_error in rot1_errors:
        for rot2_error in rot2_errors:
            for trans_error in trans_errors:
                x = pose[0] + (motion[2] + trans_error) * np.cos(np.radians(pose[2] + (motion[0] + rot1_error)))
                y = pose[1] + (motion[2] + trans_error) * np.sin(np.radians(pose[2] + (motion[0] + rot1_error)))
                poses3.append([x, y, theta])

unique, count = filter(poses3)
print(len(unique))
plot(poses1, poses2, poses3, count)
