import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import config
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from Discriminator import get_model as Pointnet_model
from Discriminator import get_loss as Pointnet_loss
import wandb



def train(Classifier):
    pass


def main():

    Classifier = Pointnet_model(k=2, normal_channel=False).to(config.DEVICE)
    Criterion = Pointnet_loss()
    
    if config.DEVICE == 'cuda':
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    optimizer = optim.Adam(
        list(Classifier.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    '''LOAD DATA'''
    dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform
        )
    
    loader = DataLoader(dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    