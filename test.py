import torch
from utils import save_checkpoint, load_checkpoint
import config
from cycleGAN_models.Generator import ReconstructionNet as Generator_Fold
from modified_Discriminator import get_model as Discriminator_Point
from cycleGAN_models.Discriminator import get_model as Disc_load
import torch.optim as optim
#from torcheval.metrics import BinaryConfussionMatrix
from tools.Visualize_tool import visualize_pc
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader_dataset import PointCloudDataset
import numpy as np



def validation(gen_FM, gen_M, POINTNET_classifier, val_loader, vis_list_female, vis_list_male, shape):
    val_loop = tqdm(val_loader, leave=True)
    classifier = POINTNET_classifier.eval()
    #TF, TM, FF, FM = []
    cf_mat = dict()
    for type in ['TP','FP','FN','TN']:
        cf_mat[type] = [0 for i in range(3)]


    for idx, data in enumerate(val_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        # Generating fakes
        fake_female = gen_FM(male)[0]
        fake_male = gen_M(female)[0]
        

        # Generate cycles
        cycle_female = gen_FM(fake_male)[0]
        cycle_male = gen_M(fake_female)[0]


        for j in range(len(vis_list_female)):
            indexes = [i for i, e in enumerate(fem_ids) if e == vis_list_female[j]]
            if indexes:
                for i in indexes:
                    torch.save(female[i], f=f"./Test-pointclouds/original_female_{(fem_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")
                    torch.save(cycle_female[i], f=f"./Test-pointclouds/cycle_female_{(fem_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")
                    torch.save(fake_male[i], f=f"./Test-pointclouds/gen_from_female_{(fem_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")

        for j in range(len(vis_list_male)):
            indexes = [i for i, e in enumerate(male_ids) if e == vis_list_male[j]]
            if indexes:
                for i in indexes:
                    torch.save(male[i], f=f"./Test-pointclouds/original_male_{(male_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")
                    torch.save(cycle_male[i], f=f"./Test-pointclouds/cycle_male_{(male_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")
                    torch.save(fake_female[i], f=f"./Test-pointclouds/gen_from_male_{(male_ids[i].split('SPRING'))[1].split('.obj')[0]}_{shape}.pt")

        for j, (female, male) in enumerate(zip([female, fake_female, cycle_female],[male, fake_male, cycle_male])):

            test_points = torch.cat((female, male), 0)
            test_ids = np.concatenate((fem_ids, male_ids))
            test_targets = torch.cat((torch.ones(len(fem_ids)),torch.zeros(len(male_ids))),0)

            indices = torch.randperm(test_targets.size()[0])
            points = test_points[indices]
            ids = test_ids[indices]
            targets = test_targets[indices]
            
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            for i in range(len(pred_choice)):
                if pred_choice[i] == targets[i] == 1:
                    cf_mat['TP'][j] += 1
                elif pred_choice[i] == targets[i] == 0:
                    cf_mat['TN'][j] += 1
                elif pred_choice[i] != targets[i] == 1:
                    cf_mat['FN'][j] += 1
                elif pred_choice[i] != targets[i] == 0:
                    cf_mat['FP'][j] += 1

        #save pointclouds for visualization purposes
        
        
       

            # for i in range(len(vis_list_male)):
            #     torch.save(male, f=f"./Test-pointclouds/original_male_{i+1}_{shape}.pt")
            #     torch.save(cycle_male, f=f"./Test-pointclouds/cycle_male_{i+1}_{shape}.pt")
            #     torch.save(fake_female, f=f"./Test-pointclouds/gen_from_male_{i+1}_{shape}.pt")
        


    return cf_mat

    


def main():
    ### Initialize model and optimizer ###
    args_gen = config.get_parser_gen()
    disc_M = Disc_load(k=2, normal_channel=False).to(config.DEVICE)
    disc_FM = Disc_load(k=2, normal_channel=False).to(config.DEVICE)
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)

    #Pointnet classifier
    POINTNET_classifier = Discriminator_Point(k=2, normal_channel=False).to(config.DEVICE)

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

    opt_class = optim.Adam(
        list(POINTNET_classifier.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    ### Load model ### 
    
    

    

    load_checkpoint(
        "CLASSIFIER_MODEL_1_accuracy_0.93.pth.tar",
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
            batch_size=10,
            shuffle=False,
            pin_memory=True,
            collate_fn=config.collate_fn
            )

    # list of pointclouds we wish to visualize:
    vis_list_female = ['SPRING0380.obj','SPRING0400.obj','SPRING0469.obj']
    vis_list_male = ['SPRING0223.obj','SPRING0300.obj','SPRING0320.obj']

    for shape in ['plane', 'sphere', 'gaussian']:#,'feature_shape']:
        args_gen.shape = shape
        gen_M = Generator_Fold(args_gen).to(config.DEVICE)
        gen_FM = Generator_Fold(args_gen).to(config.DEVICE)
        load_checkpoint(
            f"MODEL_OPTS_LOSSES_{shape}_1201.pth.tar",
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
            lr=config.LEARNING_RATE,
        )
        #gen_FM, gen_M, POINTNET_classifier = gen_FM.eval(), gen_M.eval(), POINTNET_classifier.eval()

        cf_mat = validation(gen_FM, gen_M, POINTNET_classifier, val_loader, vis_list_female, vis_list_male, shape)

        for i in range(3):
            df_cm = pd.DataFrame(np.array([[cf_mat['TP'][i],cf_mat['FP'][i]],[cf_mat['FN'][i],cf_mat['TN'][i]]]), index=['Female','Male'], columns=['Female_True','Male_true'])
            plt.figure(figsize=(6,5), dpi=100)
            sns.set(font_scale = 1.1)
            ax = sns.heatmap(df_cm, annot=True, fmt='d',cmap="Blues",cbar=False, annot_kws={"size": 40})
            ax.set_xlabel("True labels", fontsize=18, labelpad=25)
            ax.xaxis.set_ticklabels(['Female', 'Male'], fontsize=20)
            ax.set_ylabel("Predicted labels", fontsize=18, labelpad=16)
            ax.yaxis.set_ticklabels(['Female', 'Male'],fontsize=20)
            # set plot title
            type_list = ['original', 'fake','cycle']
            ax.set_title(f"Classifiers predictions on {type_list[i]} test data", fontsize=20, pad=20)
            plt.show()

        
        

if __name__ == "__main__":
    main()


# for params in gen_M.parameters():
#     g = params

# with open('readme1.txt', 'w') as f:
#     f.write(str(g))
