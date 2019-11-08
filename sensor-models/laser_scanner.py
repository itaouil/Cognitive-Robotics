"""
    Generates laser scan data
    like with opening of 250 deg,
    resolution of 2deg and maximum
    range of 12m.
"""

import cv2 as cv
import numpy as np

# Parameters
opening = 250
pixel_size = 4
resolution = 2
pose = [100,175,0]
max_range = 1200

def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Compute euclidean distance
def get_points_distance(p1, p2):
    return ((((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)) ** 0.5) * 4

def get_measurement(pose, map):
    # Iterate over max
    # range of pixels
    x, y = 0,0

    for p in range(max_range//pixel_size):
        # Compute new x and y
        x_t = int(pose[0] + (p * pixel_size) * np.cos(np.radians(pose[2])))
        y_t = int(pose[1] + (p * pixel_size) * np.sin(np.radians(pose[2])))

        # Check if [y,x] is an obstacle
        # in the map
        if (map[y_t][x_t] == 0):
            # return p * pixel_size
            return [x, y]
        else:
            x = x_t
            y = y_t

    # return max_range
    return [x, y]

def get_laser_values(pose, map, laser_angles):
    # Laser measurements
    measurements = []

    # Iterate over laser angles
    for angle in laser_angles:
        # Compute map angle
        map_angle = pose[2] + angle

        # Update measurement map
        # measurements.append([get_measurement([pose[0], pose[1], map_angle], map), map_angle])
        measurements.append(get_measurement([pose[0], pose[1], map_angle], map))

    return measurements

def run():
    # Load map
    map = cv.imread("./Assignment_04_Grid_Map.png", 0)

    # Get all possible angles
    laser_angles = [x for x in range(-124, 126, 2)]

    # Get map measurements
    # for given pose
    measurements = get_laser_values(pose, map, laser_angles)

    # Draw laser ranges on the map
    for m in measurements:
        cv.line(map, (pose[0], pose[1]), (m[0], m[1]), (100,140,40), 2)

    display_image("Laser values", map)

run()
