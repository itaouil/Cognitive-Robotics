import math

import numpy as np
import cv2

max_dist_cm = 1200  # cm
max_dist_pxl = max_dist_cm // 4  # pixel
angle = 250
half_angle = (angle // 2)


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_dist(pos1, pos2):
    """
    Return real distance by Euclidean distance.
    :param pos1:
    :param pos2:
    :return:
    """
    eucl_dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return eucl_dist


def laser_detection(pose, rotation, trans):
    """

    :param pose:
    :param rotation:
    :param step:
    :return:
    """
    return np.array([
        pose[0] + round(trans * np.cos(np.radians(pose[2] + rotation))),
        pose[1] + round(trans * np.sin(np.radians(pose[2] + rotation))),
        rotation
    ], dtype=np.int)


def detect_laser(image_map, pose_robot, pose_current):
    dist = get_dist(pose_robot, pose_current)

    if image_map[pose_current[1], pose_current[0]] == 0:
        # detected an obstacle
        return False

    if dist >= max_dist_pxl:
        # laser out of range
        return False

    return True


def display_measurements(img_map, measur):
    img_measur = img_map.copy()
    # color laser detection
    for k in range(measur.shape[0]):
        for i in range(measur.shape[1]):
            pose = measur[k, i]
            if (pose == -1).any():
                break
            img_measur[pose[1], pose[0]] = 100

    # color the initial pose with a gray value
    display_image('', img_measur)


def get_final_measurement(measur):
    final_measur = np.zeros((measur.shape[0], 3), dtype=np.int)
    for k in range(measur.shape[0]):
        val = np.array([0, 0, 0])
        for i in range(measur.shape[1]):
            pose = measur[k, i]
            if (pose == -1).any():
                break
            val = pose
        final_measur[k, :] = val
    return final_measur


def scan_environment(img_map, pose_robot):
    measur_path = np.zeros((half_angle + 1, max_dist_pxl * 2, 3), dtype=np.int)
    measur_path = np.where(measur_path == 0, -1, measur_path)

    for idx_rot, rot_cur in enumerate(range(-half_angle, half_angle + 1, 2)):
        trans = 0
        laser_pos = pose_robot.copy()
        laser_pos[2] = pose_robot[2] + rot_cur
        while detect_laser(img_map, pose_robot, laser_pos):
            measur_path[idx_rot, trans] = laser_pos
            laser_pos = laser_detection(pose_robot, rot_cur, trans)
            trans += 1

    display_measurements(img_map, measur_path)
    return get_final_measurement(measur_path)


def get_min_dist(pose_robot, measur):
    dist_min = 0
    for k in range(measur.shape[0]):
        dist_measur = get_dist(pose_robot, measur[k])
        if dist_min == 0 or dist_measur < dist_min:
            dist_min = dist_measur
    return dist_min


def compute_prob_dist(dist, sigma):
    return ((1 / np.sqrt(2 * np.pi) * sigma) *
            np.exp(-((dist ** 2) / (2 * sigma ** 2))))


def image_normalizer(img):
    return (img - img.min()) / (img.max() - img.min()) * 255


def create_distance_grid(img_map):
    sigma = 35 / 4
    img_dist = cv2.distanceTransform(img_map, cv2.DIST_L2, 5)
    img_distance_grid = img_dist.copy()
    for y in range(img_dist.shape[0]):
        for x in range(img_dist.shape[1]):
            dist = img_dist[y, x]
            img_distance_grid[y, x] = compute_prob_dist(dist, sigma)

    img_distance_grid = image_normalizer(img_distance_grid).astype(np.uint8)
    return img_distance_grid


def is_inside_map(x, y, shape):
    if 0 > y or y >= shape[0]:
        return False
    if 0 > x or x >= shape[1]:
        return False
    return True


def likelihood_pose(pose_robot, img_distance, measur):
    q = 1
    shape_img = img_distance.shape
    z_hit, z_short, z_rand, z_max = .9, 0, 0, .2
    for k in range(measur.shape[0]):
        pose_measur = measur[k]
        dist_measur = get_dist(pose_robot, pose_measur)
        if dist_measur < max_dist_pxl:
            x = int(round(pose_robot[0] + dist_measur * np.cos(np.radians(pose_robot[2] + pose_measur[2]))))
            y = int(round(pose_robot[1] + dist_measur * np.sin(np.radians(pose_robot[2] + pose_measur[2]))))
            if is_inside_map(x, y, shape_img):
                prob = (z_hit * img_distance[y, x])
            else:
                return 0
            q *= prob

    return q


def draw_likelihoods(img, img_likely):
    for y in range(img_likely.shape[0]):
        for x in range(img_likely.shape[1]):
            if img_likely[y, x] < 255 and img[y, x] != 0:
                img[y, x] = img_likely[y, x]
    return img


def create_likelihood_poses(pose_robot, img_map, img_distance, measur):
    img_likely = img_distance.copy()
    img_likely = np.where(img_likely, 0, 0).astype(np.float)
    x_init, y_init = pose_robot[:2]
    for y in range(img_map.shape[0]):
        print(y)
        for x in range(img_map.shape[1]):
            pose_in_map = pose_robot
            pose_in_map[0], pose_in_map[1] = x, y
            likely = likelihood_pose(pose_in_map, img_distance, measur)
            img_likely[y, x] = likely

    img_likely = image_normalizer(img_likely)
    img_likely = (255 - img_likely).astype(np.uint8)
    img_map = draw_likelihoods(img_map, img_likely)
    return img_map


def main_exercises():
    pose = np.array([
        375,  # x (375)
        183,  # y (183)
        210,  # rot
    ], dtype=np.int)
    img_map = cv2.imread('Assignment_04_Grid_Map.png', 0)

    measur = scan_environment(img_map, pose)

    img_distance = create_distance_grid(img_map)
    display_image('', img_distance)

    img_likely = create_likelihood_poses(pose, img_map, img_distance, measur)
    display_image('', img_likely)


if __name__ == '__main__':
    main_exercises()
