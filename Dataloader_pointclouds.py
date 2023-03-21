

import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


class PointCloudDataset(Dataset):
    def __init__(self):

        males = []
        females =[]

        #load females
        for obj in glob.iglob("./data/female/*.obj"):
            mesh = o3d.io.read_triangle_mesh(obj)

            #cetralises each mesh object
            vertices = np.asarray(mesh.vertices)
            center = vertices.mean(axis=0)
            vertices -= center
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            #converts mesh to pcd and downsamples to 2048 points
            pcd = mesh.sample_points_uniformly(2048)
            females.append(pcd)
        
        #load males
        for obj in glob.iglob("./data/male/*.obj"):
            mesh = o3d.io.read_triangle_mesh(obj)

            #cetralises each mesh object
            vertices = np.asarray(mesh.vertices)
            center = vertices.mean(axis=0)
            vertices -= center
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            #converts mesh to pcd and downsamples to 2048 points
            pcd = mesh.sample_points_uniformly(2048)
            males.append(pcd)

        
        
        self.females = females
        self.males = males
        self.pcds = np.concatenate((females, males), axis=0)
    
    def __len__(self):
        return len(self.pcds)
    
    def __getitem__(self, idx):
        return self.pcds[idx]





#Load entire dataset and stash in the pytorch class:
pcds_dataset = PointCloudDataset()

print(len(pcds_dataset))