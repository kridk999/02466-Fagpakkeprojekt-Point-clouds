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
    name = 'Classifier_run_monday_mseloss',
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
    test_loop = tqdm(loader)

    for j, data in enumerate(test_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']

        test_points = torch.cat((female, male), 0)
        test_ids = np.concatenate((fem_ids, male_ids))
        test_targets = torch.cat((torch.ones(len(fem_ids)),torch.zeros(len(male_ids))),0)

        indices = torch.randperm(test_targets.size()[0])
        test_points = test_points[indices]
        test_ids = test_ids[indices]
        test_targets = test_targets[indices]

        if config.DEVICE == 'cuda':
            test_points, test_targets = test_points.cuda(), test_targets.cuda()

        pred, _ = classifier(test_points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(test_targets.long().cpu()):
            classacc = pred_choice[test_targets == cat].eq(test_targets[test_targets == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(test_points[test_targets == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(test_targets.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(test_points.size()[0]))

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
        points = train_points[indices]
        targets = train_targets[indices]

        optimizer.zero_grad()

        train_reverse = targets.long().cpu().numpy()
        train_reverse = np.where((train_reverse==0)|(train_reverse==1), train_reverse^1, train_reverse)

        target = np.concatenate((np.expand_dims(train_reverse, axis=0),targets.unsqueeze(0),),axis=0)
        target = torch.from_numpy(target)

        if not config.DEVICE == 'cpu':
            points, target, targets = points.cuda(), target.cuda(), targets.cuda()

        pred, trans_feat = Classifier(points)
        loss = Criterion(pred.transpose(-2,1),target.float())

        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(targets.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()



        # if config.DEVICE == 'cuda':
        #     train_points, train_targets = train_points.cuda(), train_targets.cuda()

        # pred, trans_feat = Classifier(train_points)
        # train_reverse = train_targets.long().cpu().numpy()
        # train_reverse = np.where((train_reverse==0)|(train_reverse==1), train_reverse^1, train_reverse)

        # train_target = np.concatenate((np.expand_dims(train_reverse, axis=0),train_targets.unsqueeze(0),),axis=0)
        # loss = Criterion(pred, torch.tensor(train_target).transpose(0,1).float())
        # pred_choice = pred.data.max(1)[1]
    
        # correct = pred_choice.eq(train_targets.long().data).cpu().sum()


        # mean_correct.append(correct.item() / float(train_points.size()[0]))

        # #Update the optimizer for the discriminator
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    return np.mean(mean_correct)



def main():

    Classifier = Pointnet_model(k=2, normal_channel=False).to(config.DEVICE)
    Criterion = nn.MSELoss()
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
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

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
            batch_size=32,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    
    val_dataset = PointCloudDataset(
        root_female=config.VAL_DIR + "/female_test",
        root_male=config.VAL_DIR + "/male_test",
        transform=False
    )

    val_loader = DataLoader(val_dataset,
            batch_size=3,
            shuffle=False,
            pin_memory=True,
            collate_fn=config.collate_fn
            )
    best_instance_acc = 0.0
    best_class_acc = 0.0
    EPOCHS = 1
    for epoch in range(EPOCHS):
        #scheduler.step()
        Classifier = Classifier.train()
        acc = train(Classifier, Criterion, optimizer, loader)
        wandb.log({'epoch':epoch+1, 'train_accuracy':acc})
        print(f'the accuracy for epoch {epoch+1} is {acc}')
        
        if epoch % 1 == 0:
            with torch.no_grad():
                instance_acc, class_acc = test(Classifier.eval(), val_loader, num_class=2)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
                wandb.log({'test instance accuracy': instance_acc, 'Class accuracy': class_acc})

                if (instance_acc >= best_instance_acc):
                    print('Save model...')
                    filename=f"CLASSIFIER_MODEL_{epoch+1}_accuracy_{instance_acc}.pth.tar"
                    #savepath = str(checkpoints_dir) + '/best_model.pth'
                    print(f'Saving as {filename} with an instance accuracy of {instance_acc}')
                    save_checkpoint(epoch=epoch, models=[Classifier],optimizers=[optimizer], losses=acc, filename=filename)

        
            
    wandb.finish()
        
if __name__ == '__main__':
    main()