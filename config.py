import torch
import numpy as np
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
SAMPLE_POINTS = 2048
BATCH_SIZE = 1
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
