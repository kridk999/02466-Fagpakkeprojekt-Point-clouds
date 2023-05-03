import torch
import numpy as np
import argparse
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMMY_TRAIN_DIR = "data/dummy"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
FURTHEST_DISTANCE = 1.1048446043276023
SAMPLE_POINTS = 2025
BATCH_SIZE = 20
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 1
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = False
RETURN_LOSS = True
CHECKPOINT_ALL = "MODEL_OPTS_LOSSES.pth.tar"
# CHECKPOINT_GEN_M = "genM.pth.tar"
# CHECKPOINT_GEN_FM = "genFM.pth.tar"
# CHECKPOINT_CRITIC_M = "discM.pth.tar"
# CHECKPOINT_CRITIC_FM = "discFM.pth.tar"

'''
WANDB variables:
'''
project = 'training_loop_test'
user = 'fagprojekt_pointclouds'
display_name = 'training loop test'


def transform(female, male):
    if np.random.uniform(0,1) < 0.5:
        for pcl in (female, male):
            pcl *= np.random.uniform(0.7,1)
    return female, male

def collate_fn(batch):
    pc_female = [b["f_pcs"] for b in batch]
    pc_female = torch.stack(pc_female).transpose(1, 2)
    pc_male = [b["m_pcs"] for b in batch]
    pc_male = torch.stack(pc_male).transpose(1, 2)
    female_ids = [b["id_female"] for b in batch]
    male_ids = [b["id_male"] for b in batch]
    return dict(pc_female=pc_female,pc_male=pc_male, f_id = female_ids, m_id = male_ids)

def get_parser_gen():
    parser = argparse.ArgumentParser(description='FoldingNet as Generator')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    # parser.add_argument('--task', type=str, default='reconstruct', metavar='N',
    #                     choices=['reconstruct', 'classify'],
    #                     help='Experiment task, [reconstruct, classify]')
    # parser.add_argument('--encoder', type=str, default='foldingnet', metavar='N',
    #                     choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
    #                     help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')

    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=16, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='plane', metavar='N',
                        choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--dataset', type=str, default='dataset', metavar='N',
                        choices=['dataset','dummy_dataset'],
                        help='Encoder to use, [dataset, dummy_dataset]')
    # parser.add_argument('--use_rotate', action='store_true',
    #                     help='Rotate the pointcloud before training')
    # parser.add_argument('--use_translate', action='store_true',
    #                     help='Translate the pointcloud before training')
    # parser.add_argument('--use_jitter', action='store_true',
    #                     help='Jitter the pointcloud before training')
    # parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=NUM_WORKERS)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, metavar='N',
                        help='Number of episode to train ')
    # parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
    #                     help='Save snapshot interval ')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Enables CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=SAMPLE_POINTS,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    args = parser.parse_args()
    return args

def get_parser_disc():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size in training')
    #parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    #parser.add_argument('--epoch', default=NUM_EPOCHS, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    #parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    #parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    args = parser.parse_args()
    return args

