
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import Dataset
from config import SAMPLE_POINTS
import os


class PointCloudDataset(Dataset):

    def __init__(self, root_female="./data/train/female", root_male="./data/train/male", transform = None):
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

        self.furthest_distance = 1.1048446043276023 #calculated in notebook 
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
            female_array, male_array = self.transform(female=female_array, male = male_array)
        
        #normalize with regard to furtherst point in the whole dataset
        for pcl in (male_array, female_array):
            centroid = np.mean(pcl, axis=0)
            pcl -= centroid
            pcl /= self.furthest_distance
        
        #return a tensor pointcloud for each domain
        male_pointcloud, female_pointcloud = torch.from_numpy(male_array), torch.from_numpy(female_array)
        
        return female_pointcloud, male_pointcloud

    #Make a function that returns the normalvector for the points in a pointcloud
    

# data = PointCloudDataset()
# female, male = data[4]

def visualize3d(cloud):
    name = "random pointcloud"
    points = cloud.numpy() #takes tensor makes it into np.array [x,y,z] koordinates
    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    x,y,z = points[:,0], points[:,1], points[:,2]
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.8)


    # Creating color map
    my_cmap = plt.get_cmap('cool')#plt.get_cmap('hsv')

    # Creating plot
    sctt = ax.scatter3D(x, y, z,
                        alpha = 0.5,
                        c = (x + y + z),
                        cmap = my_cmap,
                        marker = 'o',
                        s=0.5)

    plt.title(str(name))
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    ax.view_init(elev=20,azim=-160,roll=0)
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-0.75,0.75) # Set x-axis range
    ax.set_ylim(-0.75,0.75) # Set y-axis range
    ax.set_zlim(-0.75,0.75) # Set z-axis range
    #fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

    # show plot
    plt.show()

#visualize3d(female)
#visualize3d(male)


