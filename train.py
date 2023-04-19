
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

def train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse,d_scaler, g_scaler, l1):
    real_Males = 0
    fake_Males = 0
    
    training_loop = tqdm(loader, leave=True)

    for idx, (female, male) in enumerate(training_loop):
        female = female.cuda()
        male = male.cuda()
        #Training discriminators for both the male and female domain
        with torch.cuda.amp.autocast():
            # Male discriminator
            fake_male = gen_M(female)
            D_M_real = disc_M(male)
            D_M_fake = disc_M(fake_male.detach())
            real_Males += D_M_real.mean().item()   
            fake_Males += D_M_fake.mean().item()  

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

            #Total discriminator loss
            D_loss = (D_M_loss + D_FM_loss) / 2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #Train the generators for male and female
        with torch.cuda.amp.autocast():
            #Adviserial loss for both generators
            D_M_fake = disc_M(fake_male)
            D_FM_fake = disc_FM(fake_female)
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
            loss_G_FM = mse(D_FM_fake, torch.ones_like(D_FM_fake))

            #Cycle loss
            cycle_female = gen_FM(fake_male)
            cycle_male = gen_M(fake_female)
            cycle_female_loss = l1(female, cycle_female)
            cycle_male_loss = l1(male, cycle_male)

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
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        #Save a couple pcl's:
        if idx % 300 == 0:

            pass



def main():
    args = config.get_parser()

    disc_M = Discriminator_Point().to(config.DEVICE)
    disc_FM = Discriminator_Point().to(config.DEVICE)
    gen_M = Generator_Fold(args).to(config.DEVICE)
    gen_FM = Generator_Fold(args).to(config.DEVICE)


    # using Adam as optimizer, is this correct?
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

    l1 = nn.L1Loss()
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
    if args.dataset == 'dataset':
        dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform
        )
    elif args.dataset == 'dummy_dataset':
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

    #create scalers for g and d:
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, d_scaler, g_scaler, l1)

        if config.SAVE_MODEL:
            save_checkpoint(epoch,gen_M, opt_gen, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(epoch,gen_FM, opt_gen, filename=config.CHECKPOINT_GEN_FM)
            save_checkpoint(epoch,disc_M, opt_disc, filename=config.CHECKPOINT_CRITIC_M)
            save_checkpoint(epoch,disc_FM, opt_disc, filename=config.CHECKPOINT_CRITIC_FM)



if __name__ == "__main__":
    data = PointCloudDataset()
    print([b["f_pcs"] for b in data[1]])
    breakpoint()
    print(config.collate_fn(data[1]))
    breakpoint()
    print('yes')
    #main()
     