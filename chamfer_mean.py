
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from utils import ChamferLoss
import math
import numpy as np
import scipy
from Generator import ReconstructionNet as Generator_Fold
import torch.optim as optim
from utils import load_checkpoint
from Discriminator import get_model as Discriminator_Point
from PlotSpecifikkePointclouds import visualize_pc

train_data = False
generated_data = True
shape = "gaussian"



def train_one_epoch(gen_M, gen_FM, loader):

    train_loop = tqdm(loader, leave=True)
    males = []
    females = []
    fake_females = []
    fake_males = []
    for idx, data in enumerate(train_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fake_female = gen_FM(male)[0].to(config.DEVICE)
        fake_male = gen_M(female)[0].to(config.DEVICE)
        fake_females.append(fake_female.detach())
        fake_males.append(fake_male.detach())
        males.append(male)
        females.append(female)
        # Generating fakes
    # for i in range(len(males)):
    #     fake_female = gen_FM(males[i])[0].to(config.DEVICE)
    #     fake_male = gen_M(females[i])[0].to(config.DEVICE)
    #     fake_females.append(fake_female.detach())
    #     fake_males.append(fake_male.detach())
        
    return males, females, fake_females, fake_males

def chamf_mean(pcs, chamferloss):
    if pcs[-1].size()[0] != pcs[-2].size()[0]:
        del pcs[-1]
    out = [(chamferloss(x.transpose(2,1), y.transpose(2,1))) for i, x in enumerate(pcs) for j, y in enumerate(pcs) if i != j]
    
    return torch.mean(torch.tensor(out)), torch.tensor(out)

def torch_compute_confidence_interval(data, confidence = 0.95):

    """
    Computes the confidence interval for a given survey of a data set.
    """
    n = len(data)
    mean = data.mean()
    # se: Tensor = scipy.stats.sem(data)  # compute standard error
    # se, mean: Tensor = torch.std_mean(data, unbiased=True)  # compute standard error
    se = data.std(unbiased=True) / (n**0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n - 1))
    ci = t_p * se
    return mean, ci


def main():
    args_gen = config.get_parser_gen()
    chamferloss = ChamferLoss()
    
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)
    
    load_checkpoint(
            "MODEL_OPTS_LOSSES_sphere_10.pth.tar",
            models=[gen_FM, gen_M],
            optimizers=[],
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
    
    male, female, fake_female, fake_male = train_one_epoch(gen_M, gen_FM, loader)

    mean_female, out_female = chamf_mean(female, chamferloss)
    mean_male, out_male = chamf_mean(male, chamferloss)
    CI_male = torch_compute_confidence_interval(out_male)
    CI_female =  torch_compute_confidence_interval(out_female)   
    mean_fake_female, out_fake_female = chamf_mean(fake_female, chamferloss)
    mean_fake_male, out_fake_male = chamf_mean(fake_male, chamferloss)
    CI_fake_male = torch_compute_confidence_interval(out_fake_male)
    CI_fake_female =  torch_compute_confidence_interval(out_fake_female)


if __name__ == "__main__":
    main()
    