import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import config
import numpy as np
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from Discriminator import get_model as Pointnet_model
from Discriminator import get_loss as Pointnet_loss
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project='Classifier_training',
    name = 'Classifier_testrun_1',
    entity=config.user,
    mode='online'
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": config.LEARNING_RATE,
    # "epochs": config.NUM_EPOCHS,
    # }
)



def train(Classifier, Criterion, optimizer, loader):
    train_loop = tqdm(loader, leave=True)
    mean_correct = []
    for idx, data in enumerate(train_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        pred_male, trans_feat = Classifier(male)
        pred_female, trans_feat = Classifier(female)
        
        loss_female = Criterion(pred_female, torch.ones_like(pred_female))
        loss_male = Criterion(pred_male, torch.zeros_like(pred_male))

        Class_loss = loss_female + loss_male

        pred_choice_male = pred_male.data.max(1)[1]
        pred_choice_female = pred_female.data.max(1)[1]

        correct_male = pred_choice_male.eq(0).cpu().sum()
        correct_female = pred_choice_female.eq(1).cpu().sum()

        mean_correct.append((correct_male.item()+correct_female.item()) / float(male.size()[0]+female.size()[0]))

        #Update the optimizer for the discriminator
        optimizer.zero_grad()
        Class_loss.backward()
        optimizer.step()

    return mean_correct



def main():

    Classifier = Pointnet_model(k=2, normal_channel=False).to(config.DEVICE)
    Criterion = nn.MSELoss()
    
    if config.DEVICE == 'cuda':
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    optimizer = optim.Adam(
        list(Classifier.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    '''LOAD DATA'''
    # if config.DATASET == 'dataset':
    dataset = PointCloudDataset(
        root_female=config.TRAIN_DIR + "/female",
        root_male=config.TRAIN_DIR + "/male",
        transform=config.transform
    )
    # elif config.DATASET == 'dummy_dataset':
    #     dataset = PointCloudDataset(
    #         root_female=config.DUMMY_TRAIN_DIR + "/female",
    #         root_male=config.DUMMY_TRAIN_DIR + "/male",
    #         transform=config.transform
    #     )
    
    loader = DataLoader(dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    EPOCHS = 100
    for epoch in range(EPOCHS):
        acc = train(Classifier, Criterion, optimizer, loader)
        wandb.log({'epoch':epoch+1, 'Accuracy':acc})
        #print(f'the accuracy for epoch {epoch+1} is {np.mean(acc)}')

        if epoch+1 == EPOCHS:
            save_checkpoint(epoch=epoch, models=[Classifier],optimizers=optimizer, losses=acc, filename=f"CLASSIFIER_MODEL{epoch+1}.pth.tar")
    wandb.finish()
        
if __name__ == '__main__':
    main()