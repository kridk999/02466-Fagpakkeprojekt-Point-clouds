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
    mode='disabled'
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": config.LEARNING_RATE,
    # "epochs": config.NUM_EPOCHS,
    # }
)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if config.DEVICE == 'cuda':
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc



def train(Classifier, Criterion, optimizer, loader):
    train_loop = tqdm(loader, leave=True)
    mean_correct = []

    for idx, data in enumerate(train_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        train_points = torch.cat((female, male), 0)
        train_ids = np.concatenate((fem_ids, male_ids))
        train_targets = torch.cat((torch.ones(len(fem_ids)),torch.zeros(len(male_ids))),0)

        indices = torch.randperm(train_targets.size()[0])
        train_points = train_points[indices]
        train_targets = train_targets[indices]

        if config.DEVICE == 'cuda':
            train_points, train_targets = train_points.cuda(), train_targets.cuda()

        pred, trans_feat = Classifier(train_points)
        loss = Criterion(pred, train_targets.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]
    
        correct = pred_choice.eq(train_targets.long().data).cpu().sum()


        mean_correct.append(correct.item() / float(train_points.size()[0]))

        #Update the optimizer for the discriminator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(mean_correct)



def main():

    Classifier = Pointnet_model(k=2, normal_channel=False).to(config.DEVICE)
    Criterion = Pointnet_loss()
    Classifier.apply(inplace_relu)
    
    if config.DEVICE == 'cuda':
            Classifier = Classifier.cuda()
            Criterion = Criterion.cuda()
    
    optimizer = optim.Adam(
        Classifier.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''LOAD DATA'''
    # if config.DATASET == 'dataset':
    dataset = PointCloudDataset(
        root_female=config.TRAIN_DIR + "/female",
        root_male=config.TRAIN_DIR + "/male",
        transform=config.transform
    )
    # elif config.DATASET == 'dummy_dataset':
    # dataset = PointCloudDataset(
    #     root_female=config.DUMMY_TRAIN_DIR + "/female",
    #     root_male=config.DUMMY_TRAIN_DIR + "/male",
    #     transform=config.transform
    #     )
    
    loader = DataLoader(dataset,
            batch_size=8,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    
    EPOCHS = 100
    for epoch in range(EPOCHS):
        scheduler.step()
        Classifier = Classifier.train()
        acc = train(Classifier, Criterion, optimizer, loader)
        wandb.log({'epoch':epoch+1, 'Accuracy':acc})
        print(f'the accuracy for epoch {epoch+1} is {acc}')



        if epoch+1 == EPOCHS:
            save_checkpoint(epoch=epoch, models=[Classifier],optimizers=optimizer, losses=acc, filename=f"CLASSIFIER_MODEL{epoch+1}.pth.tar")
    wandb.finish()
        
if __name__ == '__main__':
    main()