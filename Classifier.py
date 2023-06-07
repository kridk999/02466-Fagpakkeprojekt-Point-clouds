
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import config
import torch.optim as optim
from Discriminator import get_loss as loss
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import wandb
import numpy as np


# wandb.init(
#     # set the wandb project where this run will be logged
#     project=config.project,
#     name = config.display_name,
#     entity=config.user,
#     mode=config.WANDB_mode
#     # track hyperparameters and run metadata
#     # config={
#     # "learning_rate": config.LEARNING_RATE,
#     # "epochs": config.NUM_EPOCHS,
#     # }
# )



def train_one_epoch(classifier, loader, opt_disc, return_loss, save_pcl=False):
    best_G_loss = 1e10
    best_D_loss = 1e10
    D_correct = 0
    training_loop = tqdm(loader, leave=True)
    

    mean_correct = []
    for idx, data in enumerate(training_loop):
        
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']
        
        both = torch.cat((male, female), 0)
        
        

        labels = torch.cat((torch.ones([male.size()[0]]), torch.zeros([female.size()[0]])),0)
        
        preds, trans_feat = classifier(both)
        class_loss = loss()
        D_LOSS = class_loss(pred=preds, target=labels.long(), trans_feat = trans_feat)


        preds_choice = preds.data.max(1)[1]
        correct = preds_choice.eq(labels.long().data).cpu().sum()
        mean_correct.append(correct.item() / (float(male.size()[0]+female.size()[0])))
        
        train_instance_acc = np.mean(mean_correct)
        
        #Train Discriminator
        opt_disc.zero_grad()
        D_LOSS.backward()
        opt_disc.step()
       

    if return_loss:
        return train_instance_acc
    


def main():
    args_gen = config.get_parser_gen()
    #args_disc = config.get_parser_disc()

    classifier = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    
    return_loss = config.RETURN_LOSS

    opt_disc = optim.Adam(
        list(classifier.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    #load pretrained wheights from checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_ALL,
            models=[classifier],
            optimizers=[opt_disc],
            lr=config.LEARNING_RATE,
        )
    
    
    #load training dataset
    if args_gen.dataset == 'dataset':
        dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform
        )
    elif args_gen.dataset == 'dummy_dataset':
        dataset = PointCloudDataset(
            root_female=config.DUMMY_TRAIN_DIR + "/female",
            root_male=config.DUMMY_TRAIN_DIR + "/male",
            transform=config.transform
        )
    #load test dataset
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

    loader = DataLoader(dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )

    best_epoch_loss = 1e10
    for epoch in range(config.NUM_EPOCHS):
        if config.save_pointclouds:
            save_pcl = True if epoch % config.save_pointclouds == 0 else False

        if return_loss:
            
            D = train_one_epoch(classifier, loader, opt_disc, return_loss, save_pcl)
           # wandb.log({"LossD": D, "LossG": G,"Adviserial_loss": adv, "Cycle_loss": cycle, "epoch": epoch+1})
        else: train_one_epoch(classifier, loader, opt_disc, return_loss)
        models, opts = [classifier], [opt_disc]
        if config.SAVE_MODEL and return_loss:
            losses = [D]
            save_checkpoint(epoch, models, opts, losses, filename=config.CHECKPOINT_ALL)
        elif config.SAVE_MODEL: save_checkpoint(epoch, models, opts, losses=None, filename=config.CHECKPOINT_ALL)
        print(f'The best Discriminator loss for epoch {epoch+1} is {D}')
  #  wandb.finish()

if __name__ == "__main__":
    main()
   