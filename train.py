
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import config
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point

def train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, cycleloss, return_loss):
    real_Males = 0
    fake_Males = 0
    best_G_loss = 1e10
    best_D_loss = 1e10
    training_loop = tqdm(loader, leave=True)

    for idx, data in enumerate(training_loop):
        female = data['pc_female']
        male = data['pc_male']
        ids = data['ids']
        female = female.transpose(2,1).to(config.DEVICE)
        male = male.transpose(2,1).to(config.DEVICE)
        # Male discriminator
        fake_male, _ = gen_M(female)
        D_M_real, _ = disc_M(male)
        D_M_fake, _ = disc_M(fake_male.detach())
        
        #real_Males += D_M_real.mean().item()   
        #fake_Males += D_M_fake.mean().item()  
        
        
        #Calculating MSE loss 
        D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))
        D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))
        D_M_loss = D_M_real_loss + D_M_fake_loss

        #Female discriminator
        fake_female, _ = gen_FM(male)                      #generating a female from a male pointcloud
        D_FM_real, _ = disc_FM(female)                     #putting a true female from data through the discriminator
        D_FM_fake, _ = disc_FM(fake_female.detach())       #putting a generated female through the discriminator
        D_FM_real_loss = mse(D_FM_real, torch.ones_like(D_FM_real))
        D_FM_fake_loss = mse(D_FM_fake, torch.zeros_like(D_FM_fake))
        D_FM_loss = D_FM_real_loss + D_FM_fake_loss

        #Total discriminator loss
        D_loss = (D_M_loss + D_FM_loss) / 2
        
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        #Train the generators for male and female
        
        
        #Adviserial loss for both generators
        D_M_fake, _ = disc_M(fake_male)
        D_FM_fake, _ = disc_FM(fake_female)
        loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
        loss_G_FM = mse(D_FM_fake, torch.ones_like(D_FM_fake))

        #Cycle loss
        cycle_female, _ = gen_FM(fake_male)
        cycle_male, _ = gen_M(fake_female)

        # Set chamfer loss ind
        cycle_female_loss = cycleloss(female, cycle_female)
        cycle_male_loss = cycleloss(male, cycle_male)

        #Identity loss - gør det en forskel?
        # identity_female = gen_FM(female)
        # identity_male = gen_M(male)
        # identity_female_loss = l1(female, identity_female)
        # identity_male_loss = l1(male, identity_male)

        
        
        #Adding all generative losses together:
        G_loss = (
            loss_G_FM
            + loss_G_M
            + cycle_female_loss * config.LAMBDA_CYCLE
            + cycle_male_loss * config.LAMBDA_CYCLE
            #+ identity_female_loss * config.LAMBDA_IDENTITY
            #+ identity_male_loss * config.LAMBDA_IDENTITY
        )
        
        
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
       
        if G_loss < best_G_loss:
            best_G_loss = G_loss

        if D_loss < best_D_loss:
            best_D_loss = D_loss

        #Save a couple pcl's:
        if idx % 300 == 0:
            pass
    if return_loss:
        return best_D_loss, best_G_loss
    #return gen_FM, gen_M, disc_FM, disc_M, opt_gen, opt_disc, best_G_loss, best_D_loss



def main():
    args_gen = config.get_parser_gen()
    #args_disc = config.get_parser_disc()

    disc_M = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    disc_FM = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)

    return_loss = config.RETURN_LOSS
    
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

    cycleloss = Generator_Fold(args_gen).get_loss()
    mse = nn.MSELoss()

    #load pretrained wheights from checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_M,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_FM,
            gen_FM,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_M,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_FM,
            disc_FM,
            opt_disc,
            config.LEARNING_RATE,
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
        train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, cycleloss, return_loss)
        if config.SAVE_MODEL and best_G_loss < best_epoch_loss:
            save_checkpoint(epoch,gen_M, opt_gen, best_G_loss, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(epoch,gen_FM, opt_gen, best_G_loss, filename=config.CHECKPOINT_GEN_FM)
            save_checkpoint(epoch,disc_M, opt_disc, best_D_loss, filename=config.CHECKPOINT_CRITIC_M)
            save_checkpoint(epoch,disc_FM, opt_disc, best_D_loss, filename=config.CHECKPOINT_CRITIC_FM)
            best_epoch_loss = best_G_loss

    print(best_G_loss)
if __name__ == "__main__":
    main()
   