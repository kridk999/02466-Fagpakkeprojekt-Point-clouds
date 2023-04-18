import torch
import numpy as np
import argparse
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUMMY_TRAIN_DIR = "data/dummy"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
SAMPLE_POINTS = 2048
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_M = "genM.pth.tar"
CHECKPOINT_GEN_FM = "genFM.pth.tar"
CHECKPOINT_CRITIC_M = "criticM.pth.tar"
CHECKPOINT_CRITIC_FM = "criticFM.pth.tar"

def transform(female, male):
    if np.random.uniform(0,1) < 0.5:
        for pcl in (female, male):
            pcl *= np.random.uniform(0.7,1)
    return female, male

def collate_fn(batch):
    pc = [b["pcs"] for b in batch]
    pc = torch.stack(pc).transpose(1, 2)
    ids = [b["ids"] for b in batch]
    return dict(pc=pc, ids=ids)

def get_parser():
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
    parser.add_argument('--dataset', type=str, default='dummy_dataset', metavar='N',
                        choices=['dataset','dummy_dataset'],
                        help='Encoder to use, [dataset,dummy_dataset]')
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

# if __name__ == "__main__":
#     import torch
#     print(torch.version.cuda)
#     print(torch.cuda.is_available())
