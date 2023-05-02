
import random, torch, os, numpy as np
from dataloader_dataset import PointCloudDataset
from utils import visualize
import config


dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform)

female, female_id = dataset[0]["f_pcs"], dataset[0]["id_female"]

print(female)
print(female_id)

visualize(female,female_id,"female")