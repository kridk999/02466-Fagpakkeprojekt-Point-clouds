import torch
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "AladinLAMBDAtest/data/train"
VAL_DIR = "AladinLAMBDAtest/data/val"
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 7
NUM_EPOCHS = 201
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = None


'''
WANDB variables:
'''
WANDB_mode = 'online'                               # Can be 'offline or 'disabled'
project = f'HPC_TRAIN_ALLADIN_TEST' #f'HPC_RUN_{START_SHAPE}_{NUM_EPOCHS}epochs'
user = 's214609'
display_name = f'HPC_RUN_ALLADIN_TEST'

# transforms = A.Compose(
#     [
#         A.Resize(width=256, height=256),
#         A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
#         ToTensorV2(),
#     ],
#     additional_targets={"image0": "image"},
# )
