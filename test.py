import torch
from utils import save_checkpoint, load_checkpoint
import config
from Generator import ReconstructionNet as Generator_Fold
from Discriminator import get_model as Discriminator_Point
import torch.optim as optim
#from torcheval.metrics import BinaryConfussionMatrix
from PlotSpecifikkePointclouds import visualize_pc
import tqdm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader_dataset import PointCloudDataset
import numpy as np

file = config.CHECKPOINT_GEN_M

def validation(gen_FM, gen_M, POINTNET_classifier, val_loader, opt_disc, opt_gen, vis_list_female, vis_list_male):
    val_loop = tqdm(val_loader, leave=True)
    TF, TM, FF, FM = 0

    for idx, data in enumerate(val_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        # Generating fakes
        fake_female = gen_FM(male)
        fake_male = gen_M(female)

        # Generate cycles
        cycle_female = gen_FM(fake_male)
        cycle_male = gen_M(fake_female)

        # Classify fakes and cycles - female
        True_female = POINTNET_classifier(female)
        False_female = POINTNET_classifier(fake_female)
        cycle_False_female = POINTNET_classifier(cycle_female)

        # Classify fakes and cycles - male
        True_male = POINTNET_classifier(male)
        False_male = POINTNET_classifier(fake_male)
        cycle_False_male = POINTNET_classifier(cycle_male)

        
        #Calculate predictions
        if cycle_False_female[0] > 0.5:
            TF += 1
        if cycle_False_male[0] > 0.5:
            TM += 1
        elif cycle_False_female[0] <= 0.5:
            FF += 1
        elif cycle_False_male[0] <= 0.5:
            FM += 1

    # Visualize confusion matrix
    array = [[TF, FF],
             [FM, TM]]
    df_cm = pd.DataFrame(array, index=['Female','Male'], columns=['Female_True','Male_true'])
    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm,annot=True)
    plt.show()

    # Visualize chosen pointclouds
    if vis_list_female in fem_ids:
        visualize_pc(cycle_female)
    if vis_list_male in male_ids:
        visualize_pc(cycle_female)
    return array



def main():
    ### Initialize model and optimizer ###
    args_gen = config.get_parser_gen()
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)

    #Pointnet classifier
    POINTNET_classifier = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)

    opt_gen = optim.Adam(
        list(gen_FM.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_class = optim.Adam(
        list(gen_FM.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    ### Load model ### 
    
    disc_FM, disc_M, opt_disc = 0

    

    load_checkpoint(
        "POINTNET_classifier",
        models=[POINTNET_classifier],
        optimizers=[opt_class],
        lr=config.LEARNING_RATE
    )
    
    val_dataset = PointCloudDataset(
        root_female=config.VAL_DIR + "/female_test",
        root_male=config.VAL_DIR + "/male_test",
        transform=False
    )

    val_loader = DataLoader(val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True
            )

    # list of pointclouds we wish to visualize:
    vis_list_female = ['SPRING380.obj','SPRING400.obj','SPRING469.obj','SPRING600.obj','SPRING1050.obj']
    vis_list_male = ['SPRING223.obj','SPRING300.obj','SPRING320.obj','SPRING420.obj','SPRING450.obj']

    for shape in ['plane', 'sphere', 'gaussian']:
        load_checkpoint(
            f"MODEL_OPTS_LOSSES_{shape}_1000.pth.tar",
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
            lr=config.LEARNING_RATE,
        )

        cf_mat = validation(disc_FM,
                disc_M,
                gen_FM,
                gen_M,
                POINTNET_classifier,
                val_loader,
                opt_disc,
                opt_gen,
                vis_list_female,
                vis_list_male)


if __name__ == "__main__":
    main()


# for params in gen_M.parameters():
#     g = params

# with open('readme1.txt', 'w') as f:
#     f.write(str(g))
