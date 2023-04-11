

import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os


class PointCloudDataset(Dataset):

    def __init__(self, root_female="./data/female", root_male="./data/male", transform = None):
        self.root_female = root_female
        self.root_male = root_male
        self.transform = transform

        self.object_female = os.listdir(root_female)
        self.object_male = os.listdir(root_male)
        self.length_dataset = max(len(self.object_female), len(self.object_male))
        self.male_len = len(self.object_male)
        self.female_len = len(self.object_female)
        #self.object_all = np.concatenate((self.object_female, self.object_male), axis=0)

        self.furthest_distance = 1.103279724474332 #calculated in notebook 
        """"
        furthest_distance = 0

        for clouds in range(len(pcds_dataset)):
            distance = np.max(np.sqrt(np.sum(abs(np.asarray(pcds_dataset[clouds].points))**2,axis=1)))
            if distance > furthest_distance:
                furthest_distance = distance
        furthest_distance
        """

    def __len__(self):
        return self.length_dataset

    #given index, output a pointcloud as a tensor from both domains, have been normalized
    def __getitem__(self, idx):
        male_obj = [idx % self.male_len]
        female_obj = [idx % self.female_len]

        mesh = o3d.io.read_triangle_mesh(obj_path)
        pointcloud = mesh.sample_points_uniformly(number_of_points=2048)
        
        male_pointcloud, female_pointcloud = 1,1
        return male_pointcloud, female_pointcloud


    
    # def __getitem__(self, idx):

    #     if idx < len(self.object_female):
    #         obj_path = os.path.join(self.root_female, self.object_female[idx])
    #         label = 0  # Female label
    #     else: 
    #         obj_path = os.path.join(self.root_male, self.object_male[idx - len(self.object_female)])
    #         label = 1  # male label

    #     #convert obj. into pointclouds
    #     mesh = o3d.io.read_triangle_mesh(obj_path)
    #     pointcloud = mesh.sample_points_uniformly(number_of_points=2048)

    #     #normalize using furthest distance in whole dataset
    #     centroid = np.mean(pointcloud.points, axis=0)
    #     pointcloud.points -= centroid
    #     pointcloud.points /= self.furthest_distance

    #     if self.transform:
    #         pointcloud = self.transform(pointcloud)
    #     #if np.random > 05. augmentate to be developed...
    #     return pointcloud, label
    

data = PointCloudDataset()


print(len(data))
print(data[1])




