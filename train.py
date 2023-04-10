
#import of relevant modules and scripts
import torch
from Dataloader_pointclouds import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
#import foldingnet as Generator
#import Pointnet as Discriminator

DataLoader
breakpoint()

def train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse):
    real_Males = 0
    Fake_Males = 0

    training_loop = tqdm(loader, leave=True)

    for idx, (female, male) in enumerate(training_loop):
        
        #Training discriminators for both the male and female domain
        with torch.cuda.amp.autocast():
            # Male discriminator
            fake_male = gen_M(female)
            D_M_real = disc_M(male)
            D_M_fake = disc_M(fake_male.detach())
            real_Males += D_M_real.mean().item()   
            Fake_Males += D_M_fake.mean().item()  

            #Calculating MSE loss 
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
            D_M_loss = D_M_real_loss + D_M_fake_loss

            #Female discriminator
            fake_female = gen_FM(male)                      #generating a female from a male pointcloud
            D_FM_real = disc_FM(female)                     #putting a true female from data through the discriminator
            D_FM_fake = disc_FM(fake_female.detach())       #putting a generated female through the discriminator
            D_FM_real_loss = mse(D_FM_real, torch.ones_like(D_FM_real))
            D_FM_fake_loss = mse(D_FM_fake, torch.zeros_like(D_FM_fake))
            D_FM_loss = D_FM_real_loss + D_FM_fake_loss

            #Total discriminator 