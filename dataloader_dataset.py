

import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import SAMPLE_POINTS
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os


class PointCloudDataset(Dataset):

    def __init__(self, root_female="./data/female", root_male="./data/male", transform = None):
        self.root_female = root_female
        self.root_male = root_male
        self.transform = transform
        self.sample_points = SAMPLE_POINTS

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
        #get index for both domains
        male_obj = self.object_male[idx % self.male_len]
        female_obj = self.object_female[idx % self.female_len]

        #create path to the indexed objects
        male_path = os.path.join(self.root_male, male_obj)
        female_path = os.path.join(self.root_female, female_obj)
       
        #convert from .obj to torch tensor
        pcl_male = o3d.io.read_triangle_mesh(male_path).sample_points_uniformly(number_of_points=2048) 
        pcl_female = o3d.io.read_triangle_mesh(female_path).sample_points_uniformly(number_of_points=2048)
        male_array, female_array = np.asarray(pcl_male.points), np.asarray(pcl_female.points)
        
        #perform transformation / augmentation if turned on
        if self.transform:
            augmentations = self.transform(female=female_array, male = male_array)
            female_array = augmentations[female]
            male_array = augmentations[male]
        
        #normalize with regard to furtherst point in the whole dataset
        for pcl in (male_array, female_array):
            centroid = np.mean(pcl, axis=0)
            pcl -= centroid
            pcl /= self.furthest_distance
        
        #return a tensor pointcloud for each domain
        male_pointcloud, female_pointcloud = torch.from_numpy(male_array), torch.from_numpy(female_array)
        
        return female_pointcloud, male_pointcloud

    #Make a function that returns the normalvector for the points in a pointcloud


    
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

male, female = data[4]
breakpoint()
male




