
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import config
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint, ChamferLoss, visualize
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import wandb



wandb.init(
    # set the wandb project where this run will be logged
    project=config.project,
    name = config.display_name,
    entity=config.user
    # track hyperparameters and run metadata
    # config={
    # "learning_rate": config.LEARNING_RATE,
    # "epochs": config.NUM_EPOCHS,
    # }
)



def train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss, save_pcl=None):
    best_G_loss = 1e10
    best_D_loss = 1e10
    training_loop = tqdm(loader, leave=True)

    for idx, data in enumerate(training_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']
        #female = female.transpose(2,1).to(config.DEVICE)
        #male = male.transpose(2,1).to(config.DEVICE)

        '''''''''''
        DISCRIMINATORS
        '''''''''''

        # Male discriminator
        fake_male, _ = gen_M(female)
        D_M_real, _ = disc_M(male)
        D_M_fake, _ = disc_M(fake_male.detach())
        
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

        
        '''''''''
        GENERATORS
        '''''''''
        
        #Adviserial loss for both generators
        D_M_fake, _ = disc_M(fake_male)
        D_FM_fake, _ = disc_FM(fake_female)
        loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))
        loss_G_FM = mse(D_FM_fake, torch.ones_like(D_FM_fake))

        #Cycle loss
        cycle_female, _ = gen_FM(fake_male)
        cycle_male, _ = gen_M(fake_female)

        # Set chamfer loss ind
        cycle_female_loss = chamferloss(cycle_female, female)
        cycle_male_loss = chamferloss(cycle_male, male)

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

        #Update the optimizer for the generator
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
       
        if G_loss < best_G_loss:
            best_G_loss = G_loss

        if D_loss < best_D_loss:
            best_D_loss = D_loss

        #Save pointclouds for a chosen index:
        if save_pcl:
            if 'SPRING0470.obj' in male_ids:
                idx_male = male_ids.index('SPRING0470.obj')
                original_man = male[idx_male]
                female_male = fake_female[idx_male]
                cycle_man = cycle_male[idx_male]

                wandb.log({'original_male': wandb.Object3D(original_man.transpose(-2,1).cpu().numpy()),
                        'fake_female': wandb.Object3D(female_male.detach().transpose(-2,1).cpu().numpy()),
                        'cycle_male': wandb.Object3D(cycle_man.detach().transpose(-2,1).cpu().numpy())}, commit = False)
                
                root = os.listdir("./data/Saved_pointclouds/")
                m = len([i for i in root if 'male' in i]) // 3

                torch.save(original_man, f=f"./Saved_pointclouds/male_original{m}.pt")
                torch.save(female_male, f=f"./Saved_pointclouds/male_female{m}.pt")
                torch.save(cycle_man, f=f"./Saved_pointclouds/male_cycle{m}.pt")
                

            if 'SPRING1081.obj' in fem_ids:
                idx_female = fem_ids.index('SPRING1081.obj')
                original_woman = female[idx_female]
                male_female = fake_male[idx_female]
                cycle_woman = cycle_female[idx_female]

                wandb.log({'original_female': wandb.Object3D(original_woman.transpose(-2,1).cpu().numpy()),
                        'fake_male':wandb.Object3D(male_female.detach().transpose(-2,1).cpu().numpy()),
                        'cycle_female':wandb.Object3D(cycle_woman.detach().transpose(-2,1).cpu().numpy()),}, commit = False)
                
                root = os.listdir("./data/Saved_pointclouds/")
                w = len([i for i in root if 'female' in i]) // 3

                torch.save(original_woman, f=f"./Saved_pointclouds/female_original{w}.pt")
                torch.save(male_female, f=f"./Saved_pointclouds/female_male{w}.pt")
                torch.save(cycle_woman, f=f"./Saved_pointclouds/female_cycle{w}.pt")
                

    if return_loss:
        return best_D_loss, best_G_loss
    



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

    mse = nn.MSELoss()
    chamferloss = ChamferLoss()
    
    #load pretrained wheights from checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_ALL,
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
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
            D, G = train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss, save_pcl)
            wandb.log({"LossD": D, "LossG": G, "epoch": epoch+1})
        else: train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss)
        models, opts = [disc_FM, disc_M, gen_FM, gen_M], [opt_disc, opt_gen]
        if config.SAVE_MODEL and return_loss and G < best_epoch_loss:
            losses = [D, G] 
            save_checkpoint(epoch, models, opts, losses, filename=config.CHECKPOINT_ALL)
            best_epoch_loss = G
        elif config.SAVE_MODEL: save_checkpoint(epoch, models, opts, losses=None, filename=config.CHECKPOINT_ALL)
        print(f'The best Discriminator loss for epoch {epoch+1} is {D} and the Generator loss is {G}')
    wandb.finish()

if __name__ == "__main__":
    main()
   