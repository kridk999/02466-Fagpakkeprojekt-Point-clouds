
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from utils import ChamferLoss
import math
import numpy as np

def train_one_epoch(loader):

    train_loop = tqdm(loader, leave=True)
    males = []
    females = []
    for idx, data in enumerate(train_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        males.append(male)
        females.append(female)
        fem_ids = data['f_id']
        male_ids = data['m_id']

    return males, females


def chamf_mean(pcs, chamferloss):
    if pcs[-1].size()[0] != pcs[-2].size()[0]:
        del pcs[-1]
    out = [torch.sqrt(chamferloss(x.transpose(2,1), y.transpose(2,1))) for i, x in enumerate(pcs) for j, y in enumerate(pcs) if i != j]
    
    return torch.mean(torch.tensor(out)), out

def main():
    args_gen = config.get_parser_gen()
    chamferloss = ChamferLoss()
    
    #males(chamferloss)
    
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
    val_dataset = PointCloudDataset(
        root_female=config.VAL_DIR + "/female_test",
        root_male=config.VAL_DIR + "/male_test",
        transform=False
    )

    val_loader = DataLoader(val_dataset,
            batch_size=13,
            shuffle=False,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    #load test dataset
    

    loader = DataLoader(dataset,
            batch_size=2,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    
    
    
    male, female = train_one_epoch(loader)
    
    mean_male, out_male = chamf_mean(male, chamferloss)
    mean_female, out_female = chamf_mean(female, chamferloss)

    print(mean_male, mean_female)
    print(out_male, out_female)
    
        
if __name__ == "__main__":
    
    data = PointCloudDataset()
    
    main()
    