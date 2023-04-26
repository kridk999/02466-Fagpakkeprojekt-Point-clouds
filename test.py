import torch
from utils import save_checkpoint, load_checkpoint
import config
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import torch.optim as optim
from dataloader_dataset import PointCloudDataset
import numpy as np

file = config.CHECKPOINT_GEN_M

### Initialize model and optimizer ###

args_gen = config.get_parser_gen()
disc_FM = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
gen_M = Generator_Fold(args_gen).to(config.DEVICE)
opt_gen = optim.Adam(
        list(gen_M.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform)

### Load model ### 

load_checkpoint(
            file,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )

data = PointCloudDataset()

print(data[0]["f_pcs"])



# for params in gen_M.parameters():
#     g = params

# with open('readme1.txt', 'w') as f:
#     f.write(str(g))
