
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
#import tensorflow as tf
import random
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset, DataLoader


filepath = ["data/train/female", "data/train/male"]

#count how many objects there are in total
fem = []
mal = []
for obj in glob.iglob("./data/train/female/*.obj"):
    fem.append(1)
for obj in glob.iglob("./data/train/male/*.obj"):
    mal.append(1)

fem_tot, mal_tot, total = len(fem), len(mal), len(fem)+len(mal)

data_paths = filepath

pcds_females = []
pcds_males = []
pcds = [pcds_females, pcds_males]

tots = [fem_tot, mal_tot]

for data in range(len(data_paths)):
    files = []
    folder_to_view = data_paths[data]

    list = pcds[data]
    for file in os.listdir(folder_to_view):
        files.append(file)
    for i in tqdm(range(tots[data])):
        file = (folder_to_view + "/" + str(files[i]))
        mesh = o3d.io.read_triangle_mesh(file)
        vertices = np.asarray(mesh.vertices)
        center = vertices.mean(axis=0)
        vertices -= center
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.get_center()
        pcd = mesh.sample_points_uniformly(12500)
        #pcd = pcd.voxel_down_sample(voxel_size=0.5)
        list.append(pcd)


pcds_females_arr = []
pcds_males_arr = []
pcds_arr = [pcds_females_arr, pcds_males_arr]

pcds

for points in range(len(pcds)): 
    for i in tqdm(range(tots[points])):
        pcds_arr[points].append(np.asarray(pcds[points][i].points) )

furthest_distance = 0
for data in range(len(pcds_arr)):
    for clouds in range(tots[data]):
        distance = np.max(np.sqrt(np.sum(abs(pcds_arr[data][clouds])**2,axis=1)))
        if distance > furthest_distance:
            furthest_distance = distance
print(furthest_distance)