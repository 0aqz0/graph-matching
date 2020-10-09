import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import pickle
import json
import random
import math
import os

# classes_num = 6
labels = ["tree", "car", "bike", "pedestrain", "well", "light"]
colors = ["green", "black", "royalblue", "bisque", "grey", "yellow"]
total_num = 64
min_x = 0
max_x = 640
min_y = 0
max_y = 480
noise_mu = 0
noise_var = 2.0
visual_range = 160.0
visual_theta = 60.0 # degree
save_path = "./data/" + str(max_x) + "x" + str(max_y) +"/"+ str(total_num)

if not os.path.exists(save_path):
    os.makedirs(save_path)

def create_global_map():
    global_map = []
    for _ in range(total_num):
        random_x = random.random() * (max_x - min_x) + min_x
        random_y = random.random() * (max_y - min_y) + min_y
        random_label = random.randint(0, len(labels)-1)
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
    rotation = np.array([[np.cos(random_ori-math.pi/2), np.sin(random_ori-math.pi/2)],
                        [-np.sin(random_ori-math.pi/2), np.cos(random_ori-math.pi/2)]])
    
    local_map = []
    match = []
    for i, landmark in enumerate(map):
        global_pose = np.array([landmark[0], landmark[1]]).T
        global_pose = np.expand_dims(global_pose, axis=1)
        local_pose = (rotation @ (global_pose + translation))
        local_x = local_pose[0][0]
        local_y = local_pose[1][0]
        local_label = landmark[2]
        local_landmark = [local_x, local_y, local_label]
        theta = math.atan2(local_landmark[1], local_landmark[0]) / math.pi * 180.0
        if math.hypot(local_x, local_y) < visual_range and theta > 90 - visual_theta and theta < 90 + visual_theta:
            local_map.append(local_landmark)
            match.append(i)
        
    return base_pose, local_map, match


def visualize_all(global_map, local_map, base_pose):
    # global map
    plt.subplot(121)
    plt.gcf().gca().title.set_text("Global Map")
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(20)
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


def gen_dataset():

    # keypoints           list(float(x,y))
    # scores              float(1.)
    # descriptors         list(int[0,len(labels)]*total_nums)
    # matches             list(float(x,y))      
    # matching_scores     list(float(1.))
    # base_pose           float(x,y)

    # init raw data
    gm = create_global_map()
    bp, lm, match = create_local_map(gm)
    # make sure that lm is not empty
    while(len(lm)==0):
        bp, lm, match = create_local_map(gm)
    gm = add_gaussian_noise(gm)
    bp = torch.Tensor(bp[0:2])

    # base map
    gsize = len(gm)
    keypoints0 = torch.from_numpy(np.array(gm)[:, 0:2])
    scores0 = torch.ones(gsize)
    descriptors0 = torch.Tensor([[i[2]] for i in gm])
    idx = iter(range(gsize))
    matches0 = torch.Tensor([next(idx) if i in match else -1 for i in range(gsize)])
    matching_scores0 = torch.Tensor([1. if matches0[i] > -1 else 0. for i in range(gsize)])

    # match map
    lsize = len(lm)
    keypoints1 = torch.from_numpy(np.array(lm)[:, 0:2])
    scores1 = torch.ones(lsize)
    descriptors1 = torch.Tensor([[i[2]] for i in lm])
    matches1 = match
    matching_scores1 = torch.ones(lsize)
    
    data = dict({'keypoints0': keypoints0, 'keypoints1': keypoints1,
                 'scores0': scores0, 'scores1': scores1,
                 'descriptors0': descriptors0, 'descriptors1': descriptors1,
                 'matches0': matches0, 'matches1': matches1,
                 'matching_scores0': matching_scores0, 'matching_scores1': matching_scores1,
                 'base_pose': bp})

    return data


if __name__ == '__main__':
    # create global map
    # global_map = create_global_map()
    # visualize_map(global_map)
    
    # add noise
    # noise_map = add_gaussian_noise(global_map)
    # visualize_map(noise_map)

    # create local map
    # base_pose, local_map, _= create_local_map(global_map)

    # visualize all
    # visualize_all(global_map, local_map, base_pose)

    # print(gen_dataset())
    for i in range(1):
        # pickle.dump(gen_dataset(), open(os.path.join(save_path, str(i).zfill(4)+".pkl"), "wb"))
        np.save(os.path.join(save_path, str(i).zfill(4)+".npy"), gen_dataset())
