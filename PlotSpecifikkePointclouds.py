
import config as c
import dataloader_dataset
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from dataloader_dataset import PointCloudDataset
import open3d as o3d
import matplotlib.pyplot as plt
import tqdm


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
    my_cmap = plt.get_cmap('cool')#plt.get_cmap('cool')#plt.get_cmap('hsv')

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

visualizeNY(data[1042]["m_pcs"],"?","?")
# 

#print(data[528]["id_female"])







