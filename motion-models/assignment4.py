"""
    Exercise 3.4 Assignment3
    of Cognitive Robotics.
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def plot(data):
    for i in range(len(data)):
        if i < 100:
            color = "bo"
            x = data[i][0]
            y = data[i][1]
            plt.plot(x, y, color)
        elif i > 99 and i < 200:
            color = "go"
            x = data[i][0]
            y = data[i][1]
            plt.plot(x, y, color)
        else:
            color = "ro"
            x = data[i][0]
            y = data[i][1]
            plt.plot(x, y, color)

    plt.plot(5, -3, 'yo')

    # plt.xlim((-5, 15))
    # plt.ylim((-16, 3))
    # plt.gca().invert_yaxis()
    plt.show()

def sample(b):
    """
        Function to compute sample
        from a gaussian distribution.
    """
    # Sample variables
    sum = 0
    count = 0

    # Randomise seed
    rnd.seed()

    while count < 12:
        sum += rnd.uniform(-b, b)
        count += 1

    sum /= 2
    return sum

def motion_sampling(pose, odometry):
    """
        State transition function
        with sampling.

        u: odometry
        x: pose
    """
    new_pose = [0,0,0]

    eps_rot1 = sample(10)
    eps_rot2 = sample(5)
    eps_trans = sample(0.5)

    # Error model odometry
    rot1 = odometry[0] - eps_rot1
    rot2 = odometry[1] - eps_rot2
    trans = odometry[2] - eps_trans

    # Motion
    new_pose[0] = pose[0] + trans * np.cos(np.radians(pose[2] + rot1))
    new_pose[1] = pose[1] + trans * np.sin(np.radians(pose[2] + rot1))
    new_pose[2] = pose[2] + rot1 + rot2

    return new_pose

def transitions():
    """
        Three transitions function
        based on motion model 1.
    """
    # Counter
    count = 0

    # Points X and Y
    points = [[5,-3,45]]

    # Odometry
    odometry = [-20,-30,3]

    # Final X and Y
    plot_data = []

    # Three successive
    # iterations
    while count < 3:
        # Get poses to work with
        positions = points[count]

        # Temporary storage
        # to store new points
        new_poses = []

        # Apply motion to the
        # 100 poses we got
        for x in range(100):
            if count == 0:
                pose = motion_sampling(positions, odometry)
                new_poses.append(pose)
                plot_data.append(pose)
            else:
                pose = motion_sampling(positions[x], odometry)
                new_poses.append(pose)
                plot_data.append(pose)

        # Add our new poses to points
        points.append(new_poses)

        # Increase counter
        count += 1

    plot(plot_data)

transitions()
