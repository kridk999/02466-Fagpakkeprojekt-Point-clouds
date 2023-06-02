
import config as c
import dataloader_dataset
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from dataloader_dataset import PointCloudDataset
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import torch
from matplotlib import colors

def visualizeNY(cloud,id,gender):
    name = f"Visualization of {gender} pointcloud {id}"
    points = cloud.numpy() #takes tensor makes it into np.array [x,y,z] koordinates
    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    x,y,z = points[:,0], points[:,1], points[:,2]
    # Add x, y gridlines
    #ax.grid(b = True, color ='grey',
    #       linestyle ='-.', linewidth = 0.3,
    #        alpha = 0.8)
    ax.grid(False)


    # Creating color map
    my_cmap = plt.get_cmap('hsv')#plt.get_cmap('cool')#plt.get_cmap('hsv')

    # Creating plot
    sctt = ax.scatter3D(x, y, z,
                        alpha = 0.5,
                        c = (x + y + z),
                        cmap = my_cmap,
                        marker = 'o',
                        s=1.8)

    plt.title(str(name))
    #ax.set_xlabel('X-axis', fontweight ='bold')
    #ax.set_ylabel('Y-axis', fontweight ='bold')
    #ax.set_zlabel('Z-axis', fontweight ='bold')
    ax.view_init(elev=20,azim=-170,roll=0)
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-0.65,0.65) # Set x-axis range
    ax.set_ylim(-0.65,0.65) # Set y-axis range
    ax.set_zlim(-0.65,0.65) # Set z-axis range
    #fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

    #plt.axis('off')

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # show plot
    plt.show()


data = PointCloudDataset()
#print(data[0]["f_pcs"])

#FEMALES:
#SPRING1084.obj  has index: 493

#SPRING1640.obj = 528

#SPRING1157.obj  has index: 63

#MALES:
#SPRING1234.obj  has index: 5
#SPRING0513.obj  has index: 56
#SPRING1228.obj  has index: 1042


# for i in range(0,1531):
#     if data[i]["id_male"]=="SPRING1228.obj":
#         print("###################################")
#         print(f"SPRING1228.obj  has index: {i}")
#         break
#     else:
#         print(f"Not found at idx {i}")
# print(data[400]["id_female"])

#visualizeNY(data[1042]["m_pcs"],"?","?")
# 

#print(data[528]["id_female"])


def save_cloud_rgb(cloud, red, green, blue, filename):
    cloud = cloud.cpu()
    d = {'x': cloud[0],
         'y': cloud[1],
         'z': cloud[2],
         'red': red,
         'green': green,
         'blue': blue}
    cloud_pd = pd.DataFrame(data=d)
    cloud_pd[['red', 'green', 'blue']] = cloud_pd[['red', 'green', 'blue']].astype(np.uint8) 
    cloud = PyntCloud(cloud_pd)
    cloud.to_file(filename)



def color_pc(cloud):
    point_cloud = cloud.unsqueeze(1) #takes tensor makes it into np.array [x,y,z] koordinates

    # normalize x-axis of point cloud to be between 0 and 1
    normalized_x = (point_cloud[:, 0,:] - point_cloud[:, 0,:].min()) / (
                point_cloud[:, 0,:].max() - point_cloud[:, 0,:].min())
    normalized_x = 1 -normalized_x.squeeze()
    

    green = torch.Tensor([int(x*255) for x in colors.to_rgb('forestgreen')]).unsqueeze(1)#.to("cuda")  # RGB values for green
    yellow = torch.Tensor([int(x*255) for x in colors.to_rgb('gold')]).unsqueeze(1)#.to("cuda") # RGB values for yellow
    red = torch.Tensor([255,0,0]).unsqueeze(1)#.to("cuda")  # RGB values for red

    # Example of defining color ranges
    green_to_yellow = yellow - green
    yellow_to_red = red - yellow

    color_per_point = []
    for x in normalized_x:
        if x < 0.5: # green to yellow
            new_scale_value = x.item() / 0.5
            new_color = green + green_to_yellow*new_scale_value
            color_per_point.append(new_color.int().squeeze().tolist())
        else: # yellow to red
            new_scale_value = (x.item() - 0.5) * 2
            new_color = yellow + yellow_to_red * new_scale_value
            color_per_point.append(new_color.int().squeeze().tolist())
    color_per_point = np.array(color_per_point)
  
    return color_per_point




def visualize_pc(point_cloud, color_per_point):
    point_cloud = point_cloud.squeeze().cpu()
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=color_per_point, s=1.8)
    plt.show()



visualize_pc(data[1042]["m_pcs"],color_pc(data[1042]["m_pcs"]))
