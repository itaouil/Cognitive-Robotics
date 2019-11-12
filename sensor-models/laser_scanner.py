"""
    Generates laser scan data
    like with opening of 250 deg,
    resolution of 2deg and maximum
    range of 12m.
"""
import sys
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# Laser params
opening = 250
pixel_size = 4
resolution = 2
pose_start = [375,183,210]
max_range = 300

# Probs params
zhit = 1
sigma_hit = 35/4

def normalize(image):
    # Normalize between 0 and 255
    return ((image - image.min()) / (image.max() - image.min())) * 255

def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_measurement(pose, map):
    """
        Returns a single beam
        measurement given the
        original pose and the
        map.
    """
    for p in range(max_range//pixel_size):
        # Compute new x and y
        x = int(pose[0] + (p * pixel_size) * np.cos(np.radians(pose[2])))
        y = int(pose[1] + (p * pixel_size) * np.sin(np.radians(pose[2])))

        # Check if [y,x] is an obstacle
        # in the map
        if (map[y][x] == 0):
            # return p * pixel_size
            return [p, x, y]

    # return max_range
    return [max_range, x, y]

def get_laser_values(pose, map, laser_angles):
    """
        Returns a list of laser
        measurements where each
        entry has two values. The
        distance, and the angle of
        the laser beam.
    """
    # Laser measurements
    measurements = []

    # Iterate over laser angles
    for angle in laser_angles:
        # Compute map angle
        map_angle = pose[2] + angle

        # Laser measurement
        measurement = get_measurement([pose[0], pose[1], map_angle], map)

        # Add to measurement the angle
        measurement.append(angle)

        # Update measurement map
        measurements.append(measurement)

    return measurements

def get_distances(map):
    """
        Computes distance
        transform.
    """
    return cv.distanceTransform(map, cv.DIST_L2, 5)

def prob(dist, sigma):
    """
        Compute gaussian probability
        given the distance to closest
        obstacle.
    """
    return np.exp((-(dist**2)) / (2 * sigma**2)) * (1 / (np.sqrt(2*np.pi) * sigma))

def get_endpoint(pose, measurement):
    """
        Compute endpoint for laser
        beam given any pose x,y,theta.
    """
    # Compute endpoint x and y of the laser
    xz = int(pose[0] + measurement[0] * np.cos(np.radians(pose[2] + measurement[3])))
    yz = int(pose[1] + measurement[0] * np.sin(np.radians(pose[2] + measurement[3])))
    return [xz, yz]

def run():
    # Load map
    map = cv.imread("./Assignment_04_Grid_Map.png", 0)

    # Get all possible angles
    laser_angles = [angle for angle in range(-125, 126, 2)]

    # Get laser measurements for pose start
    measurements = get_laser_values(pose_start, map, laser_angles)

    # Get 2D distance transform of the map
    distance_transform = get_distances(map)
    # display_image("Distance transform", distance_transform.astype(np.uint8))

    # Draw laser ranges on the map
    scans = map.copy()
    for m in measurements:
        cv.line(scans, (pose_start[0], pose_start[1]), (m[1], m[2]), (100,140,40), 2)

    display_image("Laser values", scans)

    # Compute likelihood field
    prob_vectorized = np.vectorize(prob)
    likelihood_field = prob_vectorized(distance_transform, sigma_hit)
    print(likelihood_field)

    # Likelihood field size
    hlf, wlf = likelihood_field.shape[:2]

    # display_image("Likelihood field", normalize(likelihood_field).astype(np.uint8))

    # Define our likelihood map
    # likelihood_map = map.copy()
    likelihood_map = np.zeros((map.shape[0], map.shape[1]))

    # Get indices within map boundaries
    # where pixel is not an obstacle
    possible_poses = np.argwhere(map == 255)

    # Compute likelihood map for
    # all possible poses over all
    # possible orientations
    for i, pose in enumerate(possible_poses):
        print("Iteration ", i ," of ", len(possible_poses))
        likelihood = 1
        for measurement in measurements:
            # for theta in range(360):
            if(measurement[0] == max_range):
                continue
            else:
                # Compute new endpoint pose
                zk = get_endpoint([pose[1], pose[0], pose_start[2]], measurement)
                # zk = get_endpoint([x, y, pose_start[2]], measurement)

                # Check boundaries
                if (zk[1] > -1 and zk[1] < hlf) and (zk[0] > -1 and zk[0] < wlf):
                    # Compute likelihood
                    likelihood *= (zhit * likelihood_field[zk[1]][zk[0]])

        # Keep track of highest likelihood
        # if(likelihood > likelihood_map[pose[0]][pose[1]]):
        likelihood_map[pose[0]][pose[1]] = likelihood

    # Normalize my likelihood_map
    likelihood_map = normalize(likelihood_map)

    # Invert likelihood_map
    likelihood_map = 255 - likelihood_map

    for y in range(likelihood_map.shape[0]):
        for x in range(likelihood_map.shape[1]):
            if map[y][x] != 0 and map[y][x] != likelihood_map[y][x]:
                map[y][x] = likelihood_map[y][x]

    display_image("Likelihood map", map)

run()
