
import config
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
np.random.seed(42)


def visualizeNY(cloud,id,gender):
    name = f"Visualization of {gender} pointcloud {id}"
    points = cloud.numpy() #takes tensor makes it into np.array [x,y,z] koordinates
    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    x,y,z = points[:,0], points[:,1], points[:,2]
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
          linestyle ='-.', linewidth = 0.3,
           alpha = 0.8)
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
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    ax.view_init(elev=20,azim=-170,roll=0)
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-0.65,0.65) # Set x-axis range
    ax.set_ylim(-0.65,0.65) # Set y-axis range
    ax.set_zlim(-0.65,0.65) # Set z-axis range
    fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

    plt.axis('off')

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # show plot
    plt.show()



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
    #point_cloud = cloud.unsqueeze(1) #takes tensor makes it into np.array [x,y,z] koordinates
    if cloud.requires_grad: cloud = cloud.detach()
    # normalize x-axis of point cloud to be between 0 and 1
    normalized_x = (cloud[:, 2] - cloud[:, 2].min()) / (
                cloud[:, 2].max() - cloud[:, 2].min())
    normalized_x = 1 -normalized_x.squeeze()

    normalized_y = (cloud[:, 0] - cloud[:, 0].min()) / (
                cloud[:, 0].max() - cloud[:, 0].min())
    normalized_y = 1 -normalized_y.squeeze()
    

    green = torch.Tensor([int(x*255) for x in colors.to_rgb('forestgreen')]).unsqueeze(1)#.to("cuda")  # RGB values for green
    yellow = torch.Tensor([int(x*255) for x in colors.to_rgb('gold')]).unsqueeze(1)#.to("cuda") # RGB values for yellow
    cyan = torch.Tensor([int(x*255) for x in colors.to_rgb('cyan')]).unsqueeze(1)#.to("cuda") # RGB values for yellow
    red = torch.Tensor([255,0,0]).unsqueeze(1)#.to("cuda")  # RGB values for red
    blue = torch.Tensor([0,0,255]).unsqueeze(1)
    #yellow = torch.Tensor([0,255,0]).unsqueeze(1)

    # Example of defining color ranges
    blue_to_yellow = yellow - blue
    yellow_to_cyan = cyan - yellow
    green_to_yellow = yellow - green
    yellow_to_red = red - yellow

    color_per_point = []
    for x, y in zip(normalized_x,normalized_y):
        if x < 0.5: # green to yellow
            if y < 0.5:
                new_scale_value_x = x.item() / 0.5
                new_scale_value_y = y.item() / 0.5
                new_color = ((green + green_to_yellow*new_scale_value_x) + (blue + blue_to_yellow*new_scale_value_y))/2
                color_per_point.append(new_color.int().squeeze().tolist())
            else: 
                new_scale_value_x = x.item() / 0.5
                new_scale_value_y = (y.item() - 0.5) * 2
                new_color = ((green + green_to_yellow*new_scale_value_x)+ (yellow + yellow_to_cyan * new_scale_value_y))/2
                color_per_point.append(new_color.int().squeeze().tolist())
        else: # yellow to red
            if y < 0.5:
                new_scale_value_x = (x.item() - 0.5) * 2
                new_scale_value_y = y.item() / 0.5
                new_color = ((yellow + yellow_to_red * new_scale_value_x) + (blue + blue_to_yellow*new_scale_value_y))/2
                color_per_point.append(new_color.int().squeeze().tolist())
            else:
                new_scale_value_x = (x.item() - 0.5) * 2
                new_scale_value_y = (y.item() - 0.5) * 2
                new_color = ((yellow + yellow_to_red * new_scale_value_x) + (yellow + yellow_to_cyan * new_scale_value_y))/2
                color_per_point.append(new_color.int().squeeze().tolist())
    color_per_point = np.array(color_per_point)
  
    return color_per_point




def visualize_pc(point_cloud, visualize = False):
    color_per_point = color_pc(point_cloud)
    point_cloud = point_cloud.squeeze().cpu()
    if point_cloud.requires_grad: point_cloud = point_cloud.detach()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=color_per_point/255.0, s=1.8)
    ax.set_xlim3d(-0.75,0.75)
    ax.set_ylim3d(-0.75,0.75)
    ax.set_zlim3d(-0.75,0.75)
    if visualize:
        plt.show()
    return ax
    


#color_per_point = color_pc(data[1042]["m_pcs"])
data = PointCloudDataset()

test = data[1042]["m_pcs"]

test_cycle = torch.load(f='./Saved_pointclouds/male_cycle0_plane.pt', map_location=torch.device('cpu')) 
test_original = torch.load(f='./Saved_pointclouds/male_original0_plane.pt')

visualize_pc(test_original.transpose(-2,1),visualize=True)
