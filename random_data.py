import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import random
import math

classes_num = 6
labels = ["tree", "car", "bike", "pedestrain", "well", "light"]
colors = ["green", "black", "royalblue", "bisque", "grey", "yellow"]
total_num = 30
min_x = 0
max_x = 100
min_y = 0
max_y = 100
noise_mu = 0
noise_var = 2.0
visual_range = 20.0
visual_theta = 90.0 # degree

def create_global_map():
    global_map = []
    for _ in range(total_num):
        random_x = random.random() * (max_x - min_x) + min_x
        random_y = random.random() * (max_y - min_y) + min_y
        random_label = random.randint(0, classes_num-1)
        landmark = [random_x, random_y, random_label]
        global_map.append(landmark)

    return global_map


def visualize_map(map):
    for landmark in map:
        plt.scatter(landmark[0], landmark[1], c=colors[landmark[2]])
    plt.show()


def add_gaussian_noise(map):
    noise_map = []
    for landmark in map:
        noise = np.random.normal(noise_mu, noise_var, 2)
        new_x = landmark[0] + noise[0]
        new_y = landmark[1] + noise[1]
        new_label = landmark[2]
        landmark = [new_x, new_y, new_label]
        noise_map.append(landmark)
    
    return noise_map


def create_local_map(map):
    random_x = random.random() * (max_x - min_x) + min_x
    random_y = random.random() * (max_y - min_y) + min_y
    random_ori = random.random() * math.pi * 2 - math.pi
    base_pose = [random_x, random_y, random_ori]
    translation = np.array([-random_x, -random_y]).T
    translation = np.expand_dims(translation, axis=1)
    rotation = np.array([[np.cos(random_ori), np.sin(random_ori)],
                        [-np.sin(random_ori), np.cos(random_ori)]])
    local_map = []
    for landmark in map:
        global_pose = np.array([landmark[0], landmark[1]]).T
        global_pose = np.expand_dims(global_pose, axis=1)
        local_pose = rotation @ (global_pose + translation)
        local_x = local_pose[0]
        local_y = local_pose[1]
        local_label = landmark[2]
        local_landmark = [local_x, local_y, local_label]
        theta = math.fabs(math.atan2(local_landmark[1], local_landmark[0])) / math.pi * 180.0
        if math.hypot(local_x, local_y) < visual_range and theta <= visual_theta:
            local_map.append(local_landmark)
        
    return base_pose, local_map


def visualize_all(global_map, local_map, base_pose):
    # global map
    plt.subplot(121)
    plt.gcf().gca().title.set_text("Global Map")
    # plt.gcf().gca().set_xlabel("X(m)")
    # plt.gcf().gca().set_ylabel("Y(m)")
    for landmark in global_map:
        plt.scatter(landmark[0], landmark[1], c=colors[landmark[2]])

    # base pose
    plt.scatter(base_pose[0], base_pose[1], s=80, c='r', marker=(5, 1))
    orientation = [base_pose[0] + visual_range * math.cos(base_pose[2]), base_pose[1] + visual_range * math.sin(base_pose[2])]
    plt.plot([base_pose[0], orientation[0]], [base_pose[1], orientation[1]], c='r')
    arc = patches.Arc((base_pose[0], base_pose[1]), 2 * visual_range, 2 * visual_range, base_pose[2] * 180.0 / math.pi, -visual_theta, visual_theta, color='r', linewidth=2, fill=False)
    plt.gcf().gca().add_patch(arc)
    theta_min = [base_pose[0] + visual_range * math.cos(base_pose[2] + visual_theta / 180.0 * math.pi), base_pose[1] + visual_range * math.sin(base_pose[2] + visual_theta / 180.0 * math.pi)]
    plt.plot([base_pose[0], theta_min[0]], [base_pose[1], theta_min[1]], c='r')
    theta_max = [base_pose[0] + visual_range * math.cos(base_pose[2] - visual_theta / 180.0 * math.pi), base_pose[1] + visual_range * math.sin(base_pose[2] - visual_theta / 180.0 * math.pi)]
    plt.plot([base_pose[0], theta_max[0]], [base_pose[1], theta_max[1]], c='r')

    # local map
    plt.subplot(122)
    plt.gcf().gca().title.set_text("Local Map")
    # plt.gcf().gca().set_xlabel("X(m)")
    # plt.gcf().gca().set_ylabel("Y(m)")
    plt.scatter(0, 0, s=80, c='r', marker=(5, 1))
    for landmark in local_map:
        plt.scatter(landmark[0], landmark[1], c=colors[landmark[2]], marker=(5, 0))

    plt.show()


if __name__ == '__main__':
    # create global map
    global_map = create_global_map()
    # visualize_map(global_map)
    
    # add noise
    noise_map = add_gaussian_noise(global_map)
    # visualize_map(noise_map)

    # create local map
    base_pose, local_map = create_local_map(global_map)

    # visualize all
    visualize_all(global_map, local_map, base_pose)
