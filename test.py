import torch
from utils import save_checkpoint, load_checkpoint
import config
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from dataloader_dataset import PointCloudDataset
import numpy as np

file = config.CHECKPOINT_GEN_M

def test(val_loader):
    test_loop = tqdm(val_loader, leave=True)
    for idx, data in enumerate(test_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        


def main():
    ### Initialize model and optimizer ###
    args_gen = config.get_parser_gen()
    disc_M = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    disc_FM = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_FM.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_FM.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    ### Load model ### 
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_ALL,
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
            lr=config.LEARNING_RATE,
        )

    
    val_dataset = PointCloudDataset(
        root_female=config.VAL_DIR + "/female_test",
        root_male=config.VAL_DIR + "/male_test",
        transform=config.transform
    )

    val_loader = DataLoader(val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True
            )

    test(val_loader)



# for params in gen_M.parameters():
#     g = params

# with open('readme1.txt', 'w') as f:
#     f.write(str(g))
